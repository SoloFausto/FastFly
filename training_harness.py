"""Temporal stimulus and reward harness for FastFly training experiments.

The harness intentionally separates biological population metadata from the
training loop. It can evaluate an episode today; an optimizer or plasticity
rule can use the returned scalar reward to update model parameters externally.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class PopulationSpec:
    """A candidate training population and its biological fidelity."""

    key: str
    role: str
    label: str
    engine_group: str | None
    selector: str
    fidelity: str
    notes: str
    datasets: tuple[str, ...] = ("male-cns:v1.0",)

    @property
    def available(self) -> bool:
        return self.engine_group is not None

    def available_on(self, dataset_id: str) -> bool:
        return self.available and dataset_id in self.datasets


# Exact selectors use the official MaleCNS v1.0 annotation columns. Paired
# groups are split by sensory root side or motor soma side.
def _paired_populations(
    key: str,
    role: str,
    label: str,
    selector: str,
    fidelity: str,
    notes: str,
    *,
    side_field: str,
) -> tuple[PopulationSpec, PopulationSpec]:
    return tuple(
        PopulationSpec(
            key=f"{key}_{suffix}",
            role=role,
            label=f"{side_label} {label}",
            engine_group=f"{key}_{suffix}",
            selector=f"({selector}) & ({side_field} == '{code}')",
            fidelity=fidelity,
            notes=notes,
        )
        for suffix, side_label, code in (
            ("left", "Left", "L"),
            ("right", "Right", "R"),
        )
    )


NEURON_CATALOG: tuple[PopulationSpec, ...] = (
    *_paired_populations(
        "eye", "upstream", "retinal/optic-lobe sensory afferents",
        "superclass == 'ol_sensory'", "direct_sensory",
        "Root side identifies the eye of origin; the group remains broad "
        "across visual receptor types.",
        side_field="rootSide",
    ),
    *_paired_populations(
        "antenna_olf", "upstream", "antennal olfaction",
        "(class == 'olfactory') & (entryNerve == 'AN')", "direct_sensory",
        "Antennal olfactory receptor neurons, pooled across odor channels.",
        side_field="rootSide",
    ),
    *_paired_populations(
        "maxillary_olf", "upstream", "maxillary-palp olfaction",
        "(class == 'olfactory') & (entryNerve == 'MxLbN')",
        "direct_sensory",
        "Maxillary-palp olfactory neurons, pooled across odor channels.",
        side_field="rootSide",
    ),
    *_paired_populations(
        "antenna_mech", "upstream", "antennal mechanosensation",
        "(class == 'mechanosensory') & (entryNerve == 'AN')",
        "direct_sensory",
        "Pools antennal touch, vibration, wind, gravity, and auditory channels.",
        side_field="rootSide",
    ),
    *_paired_populations(
        "proboscis_gust", "upstream", "external proboscis gustation",
        "(class == 'gustatory') & (entryNerve == 'MxLbN')",
        "direct_sensory",
        "Labellar/taste-peg input pooled across tastants and valences.",
        side_field="rootSide",
    ),
    *_paired_populations(
        "pharyngeal_gust", "upstream", "pharyngeal gustation",
        "(class == 'gustatory') & (entryNerve in {'aPhN', 'PhN'})",
        "direct_sensory",
        "Internal pharyngeal taste input pooled across tastants and valences.",
        side_field="rootSide",
    ),
    *_paired_populations(
        "thermo", "upstream", "thermosensation",
        "class == 'thermosensory'", "direct_sensory",
        "Pooled across warm/cool response sign.",
        side_field="rootSide",
    ),
    *_paired_populations(
        "hygro", "upstream", "hygrosensation",
        "class == 'hygrosensory'", "direct_sensory",
        "Pooled across dry/moist response sign.",
        side_field="rootSide",
    ),
    *_paired_populations(
        "front_leg", "upstream", "front-leg sensory afferents",
        "(superclass == 'vnc_sensory') & (entryNerve == 'ProLN')",
        "direct_sensory",
        "Pools tactile, load, and proprioceptive channels from the prothoracic leg.",
        side_field="rootSide",
    ),
    *_paired_populations(
        "middle_leg", "upstream", "middle-leg sensory afferents",
        "(superclass == 'vnc_sensory') & (entryNerve == 'MesoLN')",
        "direct_sensory",
        "Pools tactile, load, and proprioceptive channels from the mesothoracic leg.",
        side_field="rootSide",
    ),
    *_paired_populations(
        "hind_leg", "upstream", "hind-leg sensory afferents",
        "(superclass == 'vnc_sensory') & (entryNerve == 'MetaLN')",
        "direct_sensory",
        "Pools tactile, load, and proprioceptive channels from the metathoracic leg.",
        side_field="rootSide",
    ),
    *_paired_populations(
        "wing", "upstream", "wing sensory afferents",
        "(superclass == 'vnc_sensory') & (entryNerve == 'ADMN')",
        "direct_sensory",
        "Pools wing bristle, campaniform, and other ADMN sensory channels.",
        side_field="rootSide",
    ),
    *_paired_populations(
        "haltere", "upstream", "haltere sensory afferents",
        "(superclass == 'vnc_sensory') & (entryNerve == 'DMetaN')",
        "direct_sensory",
        "Pools haltere and associated metathoracic mechanosensory channels.",
        side_field="rootSide",
    ),
    *_paired_populations(
        "notum", "upstream", "notum sensory afferents",
        "(superclass == 'vnc_sensory') & (entryNerve == 'PDMN')",
        "direct_sensory",
        "Dorsal thoracic mechanosensory input.",
        side_field="rootSide",
    ),
    *_paired_populations(
        "abdomen", "upstream", "abdominal sensory afferents",
        "(superclass == 'vnc_sensory') & entryNerve.startswith('Ab')",
        "direct_sensory",
        "Pools abdominal nerves and sensory modalities.",
        side_field="rootSide",
    ),
    PopulationSpec(
        "motor_proboscis", "downstream", "Proboscis motor output",
        "motor_proboscis",
        "(superclass == 'cb_motor') & (exitNerve == 'MxLbN')",
        "direct_motor",
        "Pools maxillary-labial motor functions; not extension alone.",
    ),
    PopulationSpec(
        "motor_pharynx", "downstream", "Pharyngeal motor output",
        "motor_pharynx",
        "(superclass == 'cb_motor') & (exitNerve == 'PhN')",
        "direct_motor",
        "Feeding and swallowing-related output pooled across actions.",
    ),
    PopulationSpec(
        "motor_antenna", "downstream", "Antennal motor output",
        "motor_antenna",
        "(superclass == 'cb_motor') & (exitNerve == 'AN')",
        "direct_motor", "Pooled across left and right antennal muscles.",
    ),
    PopulationSpec(
        "motor_occipital", "downstream", "Occipital motor output",
        "motor_eye",
        "(superclass in {'cb_motor', 'cb_efferent'}) & (exitNerve == 'ON')",
        "direct_motor",
        "The engine key remains motor_eye for compatibility; ON means occipital "
        "nerve, not eye rotation.",
    ),
    PopulationSpec(
        "motor_neck", "downstream", "Cervical-nerve motor output",
        "motor_neck",
        "(superclass == 'cb_motor') & (exitNerve == 'CvN')",
        "direct_motor", "Direct cranial motor output through the cervical nerve.",
    ),
    *_paired_populations(
        "motor_front_leg", "downstream", "front-leg motor neurons",
        "(superclass == 'vnc_motor') & (subclass == 'fl')", "direct_motor",
        "Direct prothoracic leg motor-neuron population.",
        side_field="somaSide",
    ),
    *_paired_populations(
        "motor_middle_leg", "downstream", "middle-leg motor neurons",
        "(superclass == 'vnc_motor') & (subclass == 'ml')", "direct_motor",
        "Direct mesothoracic leg motor-neuron population.",
        side_field="somaSide",
    ),
    *_paired_populations(
        "motor_hind_leg", "downstream", "hind-leg motor neurons",
        "(superclass == 'vnc_motor') & (subclass == 'hl')", "direct_motor",
        "Direct metathoracic leg motor-neuron population.",
        side_field="somaSide",
    ),
    *_paired_populations(
        "motor_wing", "downstream", "wing motor neurons",
        "(superclass == 'vnc_motor') & (subclass == 'wm')", "direct_motor",
        "Direct power and steering muscle motor-neuron population.",
        side_field="somaSide",
    ),
    *_paired_populations(
        "motor_haltere", "downstream", "haltere motor neurons",
        "(superclass == 'vnc_motor') & (subclass == 'hm')", "direct_motor",
        "Direct haltere motor-neuron population.",
        side_field="somaSide",
    ),
    *_paired_populations(
        "motor_abdomen", "downstream", "abdominal motor neurons",
        "(superclass == 'vnc_motor') & (subclass == 'ad')", "direct_motor",
        "Direct abdominal motor-neuron population.",
        side_field="somaSide",
    ),
    *_paired_populations(
        "motor_vnc_neck", "downstream", "VNC neck motor neurons",
        "(superclass == 'vnc_motor') & (subclass == 'nm')", "direct_motor",
        "Direct VNC neck-muscle motor-neuron population.",
        side_field="somaSide",
    ),
    PopulationSpec(
        "descending_left", "downstream", "Left descending commands",
        "descending_left",
        "(superclass == 'descending_neuron') & (somaSide == 'L')",
        "command_interface",
        "Brain-to-VNC command population; soma side does not define turn direction.",
    ),
    PopulationSpec(
        "descending_right", "downstream", "Right descending commands",
        "descending_right",
        "(superclass == 'descending_neuron') & (somaSide == 'R')",
        "command_interface",
        "Brain-to-VNC command population; soma side does not define turn direction.",
    ),
    PopulationSpec(
        "descending_center", "downstream", "Midline descending commands",
        "descending_center",
        "(superclass == 'descending_neuron') & (somaSide == 'M')",
        "command_interface",
        "Midline brain-to-VNC command population, not a decoded behavior.",
    ),
)


@dataclass(frozen=True)
class StimulusEvent:
    """Hold a current on one upstream population during [start, stop)."""

    population: str
    start: int
    stop: int
    amplitude: float = 0.5


@dataclass(frozen=True)
class RewardRule:
    """Reward proximity to a downstream firing-rate target in [start, stop)."""

    population: str
    start: int
    stop: int
    target_rate: float = 1.0
    weight: float = 1.0


@dataclass(frozen=True)
class TrainingScenario:
    name: str
    duration: int
    stimuli: Sequence[StimulusEvent]
    rewards: Sequence[RewardRule]


@dataclass(frozen=True)
class WindowResult:
    start: int
    stop: int
    active_stimuli: tuple[str, ...]
    downstream_rates: Mapping[str, float]
    reward: float | None


@dataclass(frozen=True)
class EpisodeResult:
    scenario: str
    reward: float
    windows: tuple[WindowResult, ...]


class TrainingEngine(Protocol):
    dataset_id: str | None
    def reset(self, seed: int | None = None) -> None: ...

    def get_population_indices(self, role: str, name: str) -> Any: ...

    def inject_stimuli(self, stimuli: Sequence[tuple[Any, float]]) -> None: ...

    def clear_stimulus(self) -> None: ...

    def step(self, n: int) -> Mapping[str, Any]: ...


class UnavailablePopulationError(ValueError):
    pass


class FlyTrainingHarness:
    """Runs temporal episodes and returns a normalized scalar reward."""

    def __init__(
        self,
        engine: TrainingEngine,
        catalog: Sequence[PopulationSpec] = NEURON_CATALOG,
    ):
        self.engine = engine
        self.catalog = {population.key: population for population in catalog}
        if len(self.catalog) != len(catalog):
            raise ValueError("population catalog keys must be unique")

    def run_episode(
        self,
        scenario: TrainingScenario,
        *,
        seed: int | None = None,
        reset: bool = True,
    ) -> EpisodeResult:
        """Run one episode; reward is the weighted target-rate similarity.

        Each rule scores ``1 - abs(observed_rate - target_rate)``. Rule weights
        and window durations determine the episode average, keeping the final
        reward in [0, 1]. Use target_rate=0 to reward suppression.
        """
        self._validate_scenario(scenario)
        if seed is not None and not reset:
            raise ValueError("seed requires reset=True")
        if reset:
            self.engine.reset(seed)

        input_indices = {
            event.population: self._resolve(event.population, "upstream")
            for event in scenario.stimuli
        }
        output_groups = {
            rule.population: self._resolve_group(rule.population, "downstream")
            for rule in scenario.rewards
        }
        boundaries = {0, scenario.duration}
        for event in scenario.stimuli:
            boundaries.update((event.start, event.stop))
        for rule in scenario.rewards:
            boundaries.update((rule.start, rule.stop))
        ordered_boundaries = sorted(boundaries)

        windows = []
        weighted_score = 0.0
        total_weight = 0.0
        try:
            for start, stop in zip(ordered_boundaries, ordered_boundaries[1:]):
                active_events = tuple(
                    event for event in scenario.stimuli
                    if event.start <= start < event.stop
                )
                self.engine.inject_stimuli([
                    (input_indices[event.population], event.amplitude)
                    for event in active_events
                ])
                metrics = self.engine.step(stop - start)
                motor_rates = metrics.get("motor_rates")
                if not isinstance(motor_rates, Mapping):
                    raise RuntimeError(
                        "engine step result must include a motor_rates mapping"
                    )

                active_rules = tuple(
                    rule for rule in scenario.rewards
                    if rule.start <= start < rule.stop
                )
                rates: dict[str, float] = {}
                window_score = 0.0
                window_weight = 0.0
                for rule in active_rules:
                    group = output_groups[rule.population]
                    if group not in motor_rates:
                        raise RuntimeError(
                            f"engine did not report downstream group {group!r}"
                        )
                    observed = float(motor_rates[group])
                    if not math.isfinite(observed) or not 0.0 <= observed <= 1.0:
                        raise RuntimeError(
                            f"invalid firing rate {observed!r} for {group!r}"
                        )
                    rates[rule.population] = observed
                    weighted = rule.weight * (stop - start)
                    score = 1.0 - abs(observed - rule.target_rate)
                    window_score += weighted * score
                    window_weight += weighted

                reward = None
                if window_weight:
                    reward = window_score / window_weight
                    weighted_score += window_score
                    total_weight += window_weight
                windows.append(WindowResult(
                    start=start,
                    stop=stop,
                    active_stimuli=tuple(
                        event.population for event in active_events
                    ),
                    downstream_rates=rates,
                    reward=reward,
                ))
        finally:
            self.engine.clear_stimulus()

        return EpisodeResult(
            scenario=scenario.name,
            reward=weighted_score / total_weight,
            windows=tuple(windows),
        )

    def _resolve(self, key: str, role: str) -> Any:
        group = self._resolve_group(key, role)
        return self.engine.get_population_indices(role, group)

    def _resolve_group(self, key: str, role: str) -> str:
        try:
            population = self.catalog[key]
        except KeyError:
            raise KeyError(f"unknown population {key!r}") from None
        if population.role != role:
            raise ValueError(
                f"population {key!r} is {population.role}, not {role}"
            )
        if population.engine_group is None:
            raise UnavailablePopulationError(
                f"population {key!r} is unavailable: {population.notes}"
            )
        dataset_id = getattr(self.engine, "dataset_id", None)
        if dataset_id is not None and dataset_id not in population.datasets:
            raise UnavailablePopulationError(
                f"population {key!r} is not defined for dataset {dataset_id!r}; "
                f"supported datasets: {', '.join(population.datasets)}"
            )
        return population.engine_group

    def _validate_scenario(self, scenario: TrainingScenario) -> None:
        if not isinstance(scenario.duration, int) or scenario.duration <= 0:
            raise ValueError("scenario duration must be a positive integer")
        if not scenario.rewards:
            raise ValueError("scenario must define at least one reward rule")
        for item in (*scenario.stimuli, *scenario.rewards):
            if (
                not isinstance(item.start, int)
                or not isinstance(item.stop, int)
                or not 0 <= item.start < item.stop <= scenario.duration
            ):
                raise ValueError(
                    "event windows must be integer [start, stop) ranges "
                    "inside the scenario"
                )
        for event in scenario.stimuli:
            if not math.isfinite(event.amplitude):
                raise ValueError("stimulus amplitudes must be finite")
        for rule in scenario.rewards:
            if not 0.0 <= rule.target_rate <= 1.0:
                raise ValueError("target rates must be in [0, 1]")
            if not math.isfinite(rule.weight) or rule.weight <= 0.0:
                raise ValueError("reward weights must be finite and positive")


def list_population_specs(role: str | None = None) -> tuple[PopulationSpec, ...]:
    if role not in (None, "upstream", "downstream"):
        raise ValueError("role must be 'upstream', 'downstream', or None")
    return tuple(
        population for population in NEURON_CATALOG
        if role is None or population.role == role
    )


if __name__ == "__main__":
    print(json.dumps([asdict(spec) | {"available": spec.available}
                      for spec in NEURON_CATALOG], indent=2))
