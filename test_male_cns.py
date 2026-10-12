import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

from download_male_cns import (
    DATASET_ID,
    build_population_groups,
    build_signed_graph,
    write_annotations,
    write_binary,
)


def annotation_frame():
    rows = [
        (10, "ol_sensory", "visual", "", "R1", "L", "", "", "", ""),
        (20, "vnc_sensory", "mechanosensory_tactile", "leg bristle",
         "front sensory", "R", "", "ProLN", "", "T1"),
        (30, "vnc_sensory", "mechanosensory_proprioceptive", "wing",
         "wing sensory", "L", "", "ADMN", "", "T2"),
        (40, "vnc_motor", "", "fl", "front motor", "", "L", "", "ProLN", "T1"),
        (50, "vnc_motor", "", "wm", "wing motor", "", "R", "", "ADMN", "T2"),
        (60, "descending_neuron", "", "", "descending", "", "L", "", "", "CG"),
        (70, "cb_motor", "", "pm", "proboscis motor", "", "L", "", "MxLbN", "SEZ"),
    ]
    return pd.DataFrame(rows, columns=(
        "bodyId", "superclass", "class", "subclass", "type", "rootSide",
        "somaSide", "entryNerve", "exitNerve", "somaNeuromere",
    )).assign(
        status="Traced",
        somaLocation=[np.array([i, i + 1, i + 2]) for i in range(len(rows))],
    )


@unittest.skipIf(pd is None, "pandas and pyarrow are optional data dependencies")
class MaleCNSConversionTests(unittest.TestCase):
    def test_population_groups_include_direct_vnc_inputs_and_outputs(self):
        sensory, motor, groups = build_population_groups(annotation_frame())

        np.testing.assert_array_equal(sensory["eye_left"], [0])
        np.testing.assert_array_equal(sensory["front_leg_right"], [1])
        np.testing.assert_array_equal(sensory["wing_left"], [2])
        np.testing.assert_array_equal(motor["motor_front_leg_left"], [3])
        np.testing.assert_array_equal(motor["motor_wing_right"], [4])
        np.testing.assert_array_equal(motor["descending_left"], [5])
        np.testing.assert_array_equal(motor["motor_proboscis"], [6])
        self.assertIn("vnc_sensory", groups)
        self.assertIn("vnc_motor", groups)
        assigned = np.concatenate(tuple(motor.values()))
        self.assertEqual(len(assigned), len(np.unique(assigned)))

    def test_signed_graph_omits_unknown_transmitters_and_external_bodies(self):
        pre, post, weights, omitted = build_signed_graph(
            body_ids=np.array([10, 20, 30], dtype=np.uint64),
            neuron_signs=np.array([1.0, -1.0, 0.0], dtype=np.float32),
            source_ids=np.array([20, 30, 99, 10], dtype=np.uint64),
            target_ids=np.array([10, 10, 10, 20], dtype=np.uint64),
            contact_counts=np.array([4, 8, 16, 2], dtype=np.float32),
        )

        np.testing.assert_array_equal(pre, [0, 1])
        np.testing.assert_array_equal(post, [1, 0])
        np.testing.assert_allclose(weights, [0.02, -0.04])
        self.assertEqual(omitted, 1)

    def test_annotation_and_binary_companions_share_dense_index_order(self):
        frame = annotation_frame().sort_values("bodyId").reset_index(drop=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nt_path = root / "nt.feather"
            pd.DataFrame({
                "body": frame["bodyId"],
                "consensus_nt": ["acetylcholine"] * len(frame),
            }).to_feather(nt_path)
            annotations_path = root / "fixture.annotations.npz"
            summary = write_annotations(frame, nt_path, annotations_path)

            with np.load(annotations_path, allow_pickle=True) as metadata:
                self.assertEqual(str(metadata["dataset_id"]), DATASET_ID)
                np.testing.assert_array_equal(metadata["root_ids"], frame["bodyId"])
                self.assertIn("motor_wing_right", metadata["body_motor_names"])

            binary_path = root / "fixture.bin"
            write_binary(
                binary_path,
                len(frame),
                np.array([0, 0, 1], dtype=np.uint32),
                np.array([1, 2, 2], dtype=np.uint32),
                np.array([0.1, 0.2, -0.3], dtype=np.float32),
            )
            with binary_path.open("rb") as stream:
                magic, version, neurons, edges = struct.unpack("<IIII", stream.read(16))
                offsets = np.fromfile(stream, dtype=np.uint32, count=neurons + 1)
            self.assertEqual((magic, version, neurons, edges),
                             (0x464C5957, 1, len(frame), 3))
            np.testing.assert_array_equal(offsets, [0, 2, 3, 3, 3, 3, 3, 3])
            self.assertEqual(summary["neurons"], len(frame))


if __name__ == "__main__":
    unittest.main()
