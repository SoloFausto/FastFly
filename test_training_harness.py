import unittest

from training_harness import (
    FlyTrainingHarness,
    RewardRule,
    StimulusEvent,
    TrainingScenario,
    UnavailablePopulationError,
    list_population_specs,
)


class FakeEngine:
    def __init__(self):
        self.dataset_id = "male-cns:v1.0"
        self.active = []
        self.calls = []
        self.reset_seeds = []
        self.clear_count = 0

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        self.active = []
        self.calls = []

    def get_population_indices(self, role, name):
        return (role, name)

    def inject_stimuli(self, stimuli):
        self.active = list(stimuli)

    def clear_stimulus(self):
        self.active = []
        self.clear_count += 1

    def step(self, n):
        self.calls.append((n, tuple(self.active)))
        proboscis_rate = 0.8 if self.active else 0.2
        wing_left_active = any(
            indices[1] == "wing_left" for indices, _ in self.active)
        wing_left_rate = 0.6 if wing_left_active else 0.0
        return {
            "motor_rates": {
                "motor_proboscis": proboscis_rate,
                "motor_pharynx": 0.1,
                "motor_wing_left": wing_left_rate,
                "motor_wing_right": 0.0,
            }
        }


class FlyTrainingHarnessTests(unittest.TestCase):
    def test_temporal_inputs_and_reward_windows(self):
        engine = FakeEngine()
        harness = FlyTrainingHarness(engine)
        scenario = TrainingScenario(
            name="odor-driven feeding",
            duration=10,
            stimuli=(
                StimulusEvent("antenna_olf_left", 0, 4, 0.2),
                StimulusEvent("antenna_olf_right", 2, 6, 0.3),
            ),
            rewards=(
                RewardRule("motor_proboscis", 2, 6, target_rate=0.8),
                RewardRule("motor_pharynx", 6, 10, target_rate=0.0),
            ),
        )

        result = harness.run_episode(scenario, seed=7)

        self.assertEqual(engine.reset_seeds, [7])
        self.assertEqual([call[0] for call in engine.calls], [2, 2, 2, 4])
        self.assertEqual(len(engine.calls[1][1]), 2)
        self.assertAlmostEqual(result.reward, 0.95)
        self.assertIsNone(result.windows[0].reward)
        self.assertEqual(result.windows[1].active_stimuli,
                         ("antenna_olf_left", "antenna_olf_right"))
        self.assertEqual(engine.active, [])
        self.assertEqual(engine.clear_count, 1)

    def test_wing_sensory_input_can_reward_direct_wing_motor_output(self):
        harness = FlyTrainingHarness(FakeEngine())
        scenario = TrainingScenario(
            name="left wing reflex",
            duration=5,
            stimuli=(StimulusEvent("wing_left", 0, 5, 0.4),),
            rewards=(
                RewardRule("motor_wing_left", 0, 5, target_rate=0.6),
                RewardRule("motor_wing_right", 0, 5, target_rate=0.0),
            ),
        )

        result = harness.run_episode(scenario, seed=11)

        self.assertEqual(result.reward, 1.0)
        self.assertEqual(
            result.windows[0].downstream_rates,
            {"motor_wing_left": 0.6, "motor_wing_right": 0.0},
        )

    def test_male_cns_population_is_rejected_on_flywire(self):
        engine = FakeEngine()
        engine.dataset_id = "flywire:v783"
        harness = FlyTrainingHarness(engine)
        scenario = TrainingScenario(
            name="dataset mismatch",
            duration=5,
            stimuli=(StimulusEvent("wing_left", 0, 5),),
            rewards=(RewardRule("motor_wing_left", 0, 5),),
        )

        with self.assertRaisesRegex(
            UnavailablePopulationError, "not defined for dataset"
        ):
            harness.run_episode(scenario)

    def test_catalog_preserves_biological_caveats(self):
        specs = {spec.key: spec for spec in list_population_specs()}

        self.assertEqual(specs["motor_occipital"].engine_group, "motor_eye")
        self.assertIn("not eye rotation", specs["motor_occipital"].notes)
        self.assertTrue(specs["motor_front_leg_left"].available)
        self.assertEqual(specs["motor_wing_right"].fidelity, "direct_motor")
        self.assertEqual(specs["front_leg_left"].fidelity, "direct_sensory")

    def test_invalid_scenario_is_rejected_before_stepping(self):
        engine = FakeEngine()
        harness = FlyTrainingHarness(engine)
        scenario = TrainingScenario(
            name="bad window",
            duration=4,
            stimuli=(StimulusEvent("eye_left", 2, 5),),
            rewards=(RewardRule("motor_proboscis", 0, 4),),
        )

        with self.assertRaisesRegex(ValueError, "inside the scenario"):
            harness.run_episode(scenario)
        self.assertEqual(engine.calls, [])


if __name__ == "__main__":
    unittest.main()
