"""
SimEngine — importable wrapper around the FastFly CuPy simulator.

Loads the annotation companion generated for the selected connectome for
biological stimuli, heatmap groups, positions, and motor readouts.

Usage (standalone test):
    python sim_engine.py --data flywire_v783.bin
    python sim_engine.py --data male_cns_v1.bin
"""

import base64
import json
import os
import time
import sys
import numpy as np

try:
    import cupy as cp
except ImportError:
    print("ERROR: CuPy not installed.  Run:  pip install cupy-cuda12x")
    sys.exit(1)

from flywire_sim import (CUDA_KERNELS, compile_kernels, load_connectome_binary,
                         generate_synthetic, quantize_weights_int8)



class SimEngine:
    """GPU-accelerated LIF simulator wrapping CuPy CUDA kernels."""

    def __init__(self, data_file=None, seed=42, annotations_file=None):
        self.data_file = os.path.abspath(data_file) if data_file else None
        self.dataset_id = "synthetic"
        self._manifest = None
        if data_file:
            self.n_neurons, self.n_synapses, offsets, targets, weights = \
                load_connectome_binary(data_file)
            stem, _ = os.path.splitext(self.data_file)
            manifest_file = stem + ".manifest.json"
            if os.path.exists(manifest_file):
                with open(manifest_file, "r", encoding="utf-8") as stream:
                    self._manifest = json.load(stream)
                self.dataset_id = str(self._manifest["dataset_id"])
                if int(self._manifest["neuron_count"]) != self.n_neurons:
                    raise ValueError(
                        "connectome manifest neuron count does not match binary")
                if int(self._manifest["synapse_count"]) != self.n_synapses:
                    raise ValueError(
                        "connectome manifest synapse count does not match binary")
            else:
                self.dataset_id = None
            if annotations_file:
                self.annotations_file = os.path.abspath(annotations_file)
            elif self._manifest and self._manifest.get("annotation_file"):
                annotation_path = str(self._manifest["annotation_file"])
                if not os.path.isabs(annotation_path):
                    annotation_path = os.path.join(
                        os.path.dirname(manifest_file), annotation_path)
                self.annotations_file = os.path.abspath(annotation_path)
            else:
                self.annotations_file = stem + ".annotations.npz"
        else:
            self.n_neurons, self.n_synapses, offsets, targets, weights = \
                generate_synthetic(seed=seed)
            self.annotations_file = (
                os.path.abspath(annotations_file) if annotations_file else None
            )

        self.kernels = compile_kernels()
        self.seed = seed
        self.current_step = 0

        # GPU arrays — connectivity
        self.d_offsets = cp.asarray(offsets)
        self.d_targets = cp.asarray(targets)
        w_int8, w_scales = quantize_weights_int8(weights, offsets, self.n_neurons)
        self.d_weights = cp.asarray(w_int8)
        self.d_weight_scales = cp.asarray(w_scales)

        # GPU arrays — neuron state
        rng = cp.random.default_rng(seed)
        self.d_voltage = rng.uniform(0.0, 0.9, self.n_neurons).astype(cp.float32)
        self.d_current = cp.zeros(self.n_neurons, dtype=cp.float32)

        # Spike bookkeeping
        self.spike_words = (self.n_neurons + 31) // 32
        self.d_spike_bits = cp.zeros(self.spike_words, dtype=cp.uint32)
        self.d_spike_idx  = cp.zeros(self.n_neurons, dtype=cp.uint32)
        self.d_num_spikes = cp.zeros(1, dtype=cp.uint32)

        # LIF parameters
        self.tau_decay   = np.float32(0.9)
        self.v_threshold = np.float32(1.0)
        self.v_reset     = np.float32(0.0)
        self.noise_amp   = np.float32(0.4)

        # Launch config
        self.BLOCK = 256
        self.PROP_BLOCK = 128
        self.MAX_PROP_BLOCKS = 2048
        self.neuron_blocks  = (self.n_neurons + self.BLOCK - 1) // self.BLOCK
        self.compact_blocks = (self.spike_words + self.BLOCK - 1) // self.BLOCK

        # Stimulus state
        self._stimulus_indices = None
        self._stimulus_amplitude = 0.0

        # Last-step spike indices (for 3D viz)
        self._last_spike_indices = np.array([], dtype=np.int32)

        # GPU accumulators for sync-free counting
        self.d_total_spikes = cp.zeros(1, dtype=cp.uint64)

        # Feature toggles (can be set at runtime)
        self.send_active_indices = True
        self.send_group_rates = True
        self.send_motor_rates = True
        self.active_indices_interval = 3  # only transfer every Nth batch
        self._batch_counter = 0

        # Load annotations
        self._load_annotations()

        # Metrics history
        self.group_rates_history = []

    def _load_annotations(self):
        """Load and validate the annotation companion for this connectome."""
        if not self.annotations_file or not os.path.exists(self.annotations_file):
            expected = self.annotations_file or "none for synthetic data"
            print(f"No matching annotation file found ({expected})")
            self._root_ids = None
            self._positions = None
            self._super_class = None
            self._setup_fallback_groups()
            self._use_annotations = False
            return

        print(f"Loading neuron annotations from {self.annotations_file}...")
        with np.load(self.annotations_file, allow_pickle=True) as data:
            annotation_dataset_id = (
                str(data["dataset_id"]) if "dataset_id" in data else None
            )
            if (
                self.dataset_id is not None
                and annotation_dataset_id is not None
                and annotation_dataset_id != self.dataset_id
            ):
                raise ValueError(
                    f"annotation dataset {annotation_dataset_id!r} does not match "
                    f"connectome dataset {self.dataset_id!r}"
                )
            if self.dataset_id is None:
                self.dataset_id = annotation_dataset_id

            if "root_ids" not in data:
                raise ValueError("annotation file does not contain root_ids")
            self._root_ids = data["root_ids"].astype(np.int64)
            if len(self._root_ids) != self.n_neurons:
                raise ValueError(
                    f"annotation neuron count {len(self._root_ids)} does not match "
                    f"connectome neuron count {self.n_neurons}"
                )

            if "pos_x" in data:
                self._positions = np.stack([
                    data["pos_x"], data["pos_y"], data["pos_z"]
                ], axis=1).astype(np.float32)
                if len(self._positions) != self.n_neurons:
                    raise ValueError("annotation position count does not match connectome")
                print(f"  3D positions loaded: {self._positions.shape}")
            else:
                self._positions = None
            self._super_class = (
                data["super_class"].copy() if "super_class" in data else None
            )

            self._stimuli = {}
            for raw_name in data["stim_names"]:
                name = str(raw_name)
                safe = (
                    "stim_" + name.replace(" ", "_").replace("/", "_")
                    .replace("(", "").replace(")", "")
                )
                if safe in data:
                    self._stimuli[name] = self._checked_indices(
                        data[safe], f"stimulus {name!r}")

            self.group_labels = [str(name) for name in data["group_names"]]
            self.num_groups = len(self.group_labels)
            self._group_indices = [
                self._checked_indices(
                    data["group_" + name], f"heatmap group {name!r}")
                for name in self.group_labels
            ]

            self._body_sensory = {}
            if "body_sensory_names" in data:
                for raw_name in data["body_sensory_names"]:
                    name = str(raw_name)
                    key = "bsens_" + name
                    if key in data:
                        self._body_sensory[name] = self._checked_indices(
                            data[key], f"upstream population {name!r}")

            self._body_motor = {}
            if "body_motor_names" in data:
                for raw_name in data["body_motor_names"]:
                    name = str(raw_name)
                    key = "bmotor_" + name
                    if key in data:
                        self._body_motor[name] = self._checked_indices(
                            data[key], f"downstream population {name!r}")

        self._neuron_to_group = cp.full(
            self.n_neurons, -1, dtype=cp.int32)
        for group, indices in enumerate(self._group_indices):
            self._neuron_to_group[cp.asarray(indices.astype(np.int64))] = group

        self._motor_group_names = list(self._body_motor)
        self._num_motor_groups = len(self._motor_group_names)
        self._neuron_to_motor = cp.full(
            self.n_neurons, -1, dtype=cp.int32)
        assigned_motor = np.zeros(self.n_neurons, dtype=np.bool_)
        for group, name in enumerate(self._motor_group_names):
            indices = self._body_motor[name]
            if assigned_motor[indices].any():
                raise ValueError(
                    f"downstream population {name!r} overlaps an earlier group")
            assigned_motor[indices] = True
            self._neuron_to_motor[cp.asarray(indices.astype(np.int64))] = group

        self._use_annotations = True
        print(
            f"  dataset {self.dataset_id or 'unknown'}: "
            f"{len(self._stimuli)} stimuli, {self.num_groups} heatmap groups"
        )
        print(
            f"  {len(self._body_sensory)} body sensory, "
            f"{len(self._body_motor)} body motor"
        )

    def _checked_indices(self, values, description):
        indices = np.asarray(values, dtype=np.int32).reshape(-1)
        if indices.size and (
            int(indices.min()) < 0 or int(indices.max()) >= self.n_neurons
        ):
            raise ValueError(f"{description} contains out-of-range neuron indices")
        return indices

    def _setup_fallback_groups(self):
        """Fallback: equal-size index-range groups."""
        self.num_groups = 20
        group_size = self.n_neurons // self.num_groups
        self.group_labels = [f"Group {i}" for i in range(self.num_groups)]
        self._group_indices = []
        for g in range(self.num_groups):
            start = g * group_size
            end = start + group_size if g < self.num_groups - 1 else self.n_neurons
            self._group_indices.append(np.arange(start, end, dtype=np.int32))

        self._neuron_to_group = cp.full(self.n_neurons, -1, dtype=cp.int32)
        for g, indices in enumerate(self._group_indices):
            self._neuron_to_group[cp.asarray(indices.astype(np.int64))] = g

        self._stimuli = {
            "Neurons 0-1000": np.arange(0, min(1000, self.n_neurons), dtype=np.int32),
        }
        self._body_sensory = {}
        self._body_motor = {}
        self._motor_group_names = []
        self._num_motor_groups = 0
        self._neuron_to_motor = cp.full(self.n_neurons, -1, dtype=cp.int32)

    def inject_stimulus(self, neuron_indices, amplitude=0.5):
        self.inject_stimuli(((neuron_indices, amplitude),))

    def inject_stimuli(self, stimuli):
        """Apply multiple population stimuli, summing amplitudes on overlap."""
        index_parts = []
        amplitude_parts = []
        for neuron_indices, amplitude in stimuli:
            indices = np.asarray(neuron_indices, dtype=np.int64).reshape(-1)
            amplitude = float(amplitude)
            if not np.isfinite(amplitude):
                raise ValueError("stimulus amplitude must be finite")
            if indices.size == 0:
                continue
            if indices.min() < 0 or indices.max() >= self.n_neurons:
                raise IndexError("stimulus neuron index is out of range")
            index_parts.append(indices)
            amplitude_parts.append(
                np.full(indices.size, amplitude, dtype=np.float32))

        if not index_parts:
            self.clear_stimulus()
            return

        indices = np.concatenate(index_parts)
        amplitudes = np.concatenate(amplitude_parts)
        order = np.argsort(indices)
        sorted_indices = indices[order]
        sorted_amplitudes = amplitudes[order]
        unique_indices, starts = np.unique(sorted_indices, return_index=True)
        summed_amplitudes = np.add.reduceat(sorted_amplitudes, starts)

        self._stimulus_indices = cp.asarray(unique_indices)
        self._stimulus_amplitude = cp.asarray(summed_amplitudes)

    def clear_stimulus(self):
        self._stimulus_indices = None
        self._stimulus_amplitude = 0.0

    def set_noise_amp(self, value):
        self.noise_amp = np.float32(value)

    def reset(self, seed=None):
        """Reset dynamic state for a reproducible training episode."""
        if seed is not None:
            self.seed = int(seed)
        rng = cp.random.default_rng(self.seed)
        self.d_voltage[...] = rng.uniform(
            0.0, 0.9, self.n_neurons).astype(cp.float32)
        self.d_current.fill(0)
        self.d_spike_bits.fill(0)
        self.d_spike_idx.fill(0)
        self.d_num_spikes.fill(0)
        self.d_total_spikes.fill(0)
        self.current_step = 0
        self._last_spike_indices = np.array([], dtype=np.int32)
        self._batch_counter = 0
        self.group_rates_history.clear()
        self.clear_stimulus()

    def get_population_indices(self, role, name):
        """Return indices for a named training input or output population."""
        if role == "upstream":
            populations = self._body_sensory
        elif role == "downstream":
            populations = self._body_motor
        else:
            raise ValueError("role must be 'upstream' or 'downstream'")
        try:
            return populations[name]
        except KeyError:
            available = ", ".join(sorted(populations)) or "none"
            raise KeyError(
                f"unknown {role} population {name!r}; available: {available}"
            ) from None

    def step(self, n=50):
        """Run n timesteps and return a metrics dict.

        ZERO per-substep GPU→CPU syncs:
        - propagate_v2 reads d_num_spikes from device memory (no CPU readback)
        - count_spikes kernel counts groups/motors/total from spike_bits on GPU
        - compact kernel still runs (needed for propagate's spike_idx array)
        - Single sync at batch end to transfer results to CPU
        - active_indices only transferred when send_active_indices is True
        """
        t_start = time.perf_counter()

        # GPU accumulators — zeroed once, accumulated across all substeps (uint64 for atomicAdd)
        d_group_counts = cp.zeros(self.num_groups, dtype=cp.uint64)
        d_motor_counts = cp.zeros(max(self._num_motor_groups, 1), dtype=cp.uint64)
        d_total_spikes = self.d_total_spikes
        d_total_spikes.fill(0)

        # Local refs to avoid Python attribute lookups in inner loop
        d_current = self.d_current
        d_voltage = self.d_voltage
        d_spike_bits = self.d_spike_bits
        d_spike_idx = self.d_spike_idx
        d_num_spikes = self.d_num_spikes
        d_offsets = self.d_offsets
        d_targets = self.d_targets
        d_weights = self.d_weights
        d_weight_scales = self.d_weight_scales
        neuron_to_group = self._neuron_to_group
        neuron_to_motor = self._neuron_to_motor
        k_update_with_noise = self.kernels["update_with_noise"]
        k_compact = self.kernels["compact"]
        k_propagate_v2 = self.kernels["propagate_v2"]
        k_count = self.kernels["count_spikes"]
        neuron_blocks = self.neuron_blocks
        compact_blocks = self.compact_blocks
        BLOCK = self.BLOCK
        PROP_BLOCK = self.PROP_BLOCK
        MAX_PROP_BLOCKS = self.MAX_PROP_BLOCKS
        n_neurons_i32 = np.int32(self.n_neurons)
        spike_words_i32 = np.int32(self.spike_words)
        stim_indices = self._stimulus_indices
        stim_amp = self._stimulus_amplitude

        for sub in range(n):
            d_num_spikes.fill(0)

            if stim_indices is not None:
                d_current[stim_indices] += stim_amp

            k_update_with_noise(
                (neuron_blocks,), (BLOCK,),
                (d_voltage, d_current, d_spike_bits,
                 n_neurons_i32, spike_words_i32,
                 self.tau_decay, self.v_threshold, self.v_reset,
                 np.uint32(self.seed), np.uint32(self.current_step),
                 self.noise_amp))

            k_compact(
                (compact_blocks,), (BLOCK,),
                (d_spike_bits, d_spike_idx, d_num_spikes,
                 spike_words_i32, n_neurons_i32))

            # Count group/motor/total spikes from spike_bits — pure GPU, no sync
            k_count(
                (compact_blocks,), (BLOCK,),
                (d_spike_bits, neuron_to_group, d_group_counts,
                 neuron_to_motor, d_motor_counts, d_total_spikes,
                 spike_words_i32, n_neurons_i32))

            # Propagate v2 — reads d_num_spikes from device memory, no CPU sync
            k_propagate_v2(
                (MAX_PROP_BLOCKS,), (PROP_BLOCK,),
                (d_spike_idx, d_num_spikes,
                 d_offsets, d_targets, d_weights, d_weight_scales, d_current))

            self.current_step += 1

        # === Single batch-end sync — all GPU work done ===
        cp.cuda.Stream.null.synchronize()

        total_spikes = int(d_total_spikes[0])
        t_elapsed = time.perf_counter() - t_start
        firing_rate = total_spikes / (n * self.n_neurons) if self.n_neurons > 0 else 0
        steps_per_sec = n / t_elapsed if t_elapsed > 0 else 0

        result = {
            "step": self.current_step,
            "spike_count": total_spikes,
            "firing_rate": round(firing_rate, 6),
            "mean_voltage": round(float(d_voltage.mean()), 4),
            "steps_per_sec": round(steps_per_sec, 1),
        }

        # Group rates (for heatmap) — only compute if enabled
        if self.send_group_rates:
            group_spike_counts = d_group_counts.get()
            group_rates = []
            for g in range(self.num_groups):
                group_n = len(self._group_indices[g])
                rate = float(group_spike_counts[g]) / (n * group_n) if group_n > 0 else 0
                group_rates.append(round(rate, 6))
            self.group_rates_history.append(group_rates)
            if len(self.group_rates_history) > 200:
                self.group_rates_history = self.group_rates_history[-200:]
            result["group_rates"] = group_rates

        # Motor rates — only compute if enabled
        if self.send_motor_rates:
            motor_spike_counts = d_motor_counts.get()
            motor_rates = {}
            for g, name in enumerate(self._motor_group_names):
                group_n = len(self._body_motor[name])
                rate = float(motor_spike_counts[g]) / (n * group_n) if group_n > 0 else 0
                motor_rates[name] = round(rate, 6)
            result["motor_rates"] = motor_rates

        # Active indices (for 3D viz) — only transfer every Nth batch
        self._batch_counter += 1
        if self.send_active_indices and self._batch_counter % self.active_indices_interval == 0:
            num_last = int(d_num_spikes[0])
            if num_last > 0:
                self._last_spike_indices = d_spike_idx[:num_last].get().astype(np.int32)
            else:
                self._last_spike_indices = np.array([], dtype=np.int32)
            result["active_indices"] = self._last_spike_indices.tolist()

        return result

    # --- Data accessors ---

    def get_predefined_stimuli(self):
        return list(self._stimuli.keys())

    def get_body_info(self):
        sensory = {k: len(v) for k, v in self._body_sensory.items()}
        motor = {k: len(v) for k, v in self._body_motor.items()}
        return {"sensory": sensory, "motor": motor}

    def get_positions_b64(self):
        """Return neuron positions as base64-encoded float32 array [N*3]."""
        if self._positions is not None:
            return base64.b64encode(self._positions.tobytes()).decode('ascii')
        return None

    def get_neuron_classes(self):
        """Return super_class per neuron for 3D coloring."""
        if self._super_class is not None:
            # Encode as int: unique classes -> color indices
            unique = sorted(set(self._super_class))
            class_to_id = {c: i for i, c in enumerate(unique)}
            ids = np.array([class_to_id.get(c, 0) for c in self._super_class],
                           dtype=np.uint8)
            return {
                "labels": unique,
                "ids_b64": base64.b64encode(ids.tobytes()).decode('ascii')
            }
        return None

    def get_motor_detail(self, group_name):
        """Return detail for a motor group: neuron indices, root_ids, active status."""
        if group_name not in self._body_motor:
            return None
        indices = self._body_motor[group_name]
        active_set = set(self._last_spike_indices.tolist())
        neurons = []
        for idx in indices:
            idx = int(idx)
            rid = int(self._root_ids[idx]) if self._root_ids is not None else idx
            neurons.append({
                "index": idx,
                "root_id": rid,
                "active": idx in active_set,
            })
        return {"group": group_name, "neurons": neurons}

    def apply_predefined_stimulus(self, name, amplitude=None):
        if name not in self._stimuli:
            return False
        indices = self._stimuli[name]
        amp = amplitude if amplitude is not None else 0.5
        self.inject_stimulus(indices, amp)
        return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Binary connectome file")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=50)
    args = parser.parse_args()

    engine = SimEngine(data_file=args.data)
    print(f"\nSimEngine ready: {engine.n_neurons} neurons, {engine.n_synapses} synapses")
    print(f"Stimuli: {engine.get_predefined_stimuli()}")
    print(f"Groups:  {engine.group_labels}")
    print(f"Positions: {'yes' if engine._positions is not None else 'no'}")
    print(f"Running {args.steps} steps in batches of {args.batch}...\n")

    for i in range(0, args.steps, args.batch):
        metrics = engine.step(n=args.batch)
        print(f"  Step {metrics['step']:>6d}  "
              f"spikes={metrics['spike_count']:>6d}  "
              f"rate={metrics['firing_rate']*100:>5.2f}%  "
              f"V_mean={metrics['mean_voltage']:.3f}  "
              f"active_3d={len(metrics.get('active_indices', []))}  "
              f"steps/s={metrics['steps_per_sec']:.0f}")

    print("\nDone.")
