# FastFly

GPU-accelerated simulator for the seamless adult male *Drosophila melanogaster* central nervous system connectome—brain and ventral nerve cord—using 165,122 traced neurons and approximately 24.5 million signed modeled connections on one consumer NVIDIA GPU.

![Fruit fly brain](fruitfly.jpg)

## How it works

- **Neuron model:** Leaky Integrate-and-Fire (LIF), validated by [arXiv:2404.17128](https://arxiv.org/abs/2404.17128)
- **Connectivity:** CSR sparse format with FP16 synaptic weights
- **Spike propagation:** Push model — only processes synapses of neurons that actually fired
- **Spike detection:** Warp ballot intrinsics for bit-packed spike flags
- **Load balancing:** Warp-per-spike with grid-stride loop

## Data sources

### MaleCNS v1.0: brain and ventral nerve cord

The primary dataset is Janelia's seamless [MaleCNS v1.0](https://male-cns.janelia.org/)
connectome. Unlike a stitched brain/MANC model, it contains the intact brain,
neck connective, and ventral nerve cord in one reconstruction and one neuron-ID
space.

- **Official download:** [male-cns.janelia.org/download](https://male-cns.janelia.org/download/)
- **Dataset:** `male-cns:v1.0`, licensed CC-BY 4.0
- **Imported universe:** 165,122 neurons with annotation status `Traced`
- **Connectivity:** official segment-to-segment aggregate contact table
- **Annotations:** classes, types, sides, entry/exit nerves, neuromeres, and soma positions
- **Functional sign approximation:** acetylcholine is positive; GABA, glutamate,
  and histamine are negative. Neuromodulators and unresolved transmitters are
  omitted instead of being assigned a made-up fast synaptic polarity.

`download_male_cns.py` downloads the official flat Feather tables, builds the
CSR simulator binary, and emits the matching annotation companion. The
converter preserves direct leg, wing, haltere, abdominal, and cranial sensory
and motor populations.

### FlyWire v783: brain-only alternative

The original adult female brain-only path remains available:

- **Connectivity:** `Connectivity_783.parquet` and `Completeness_783.csv`
- **Repository:** [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model)
- **Annotations:** [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations)
- **Contents:** 139,255 neurons and 54.5 million signed connections
- **Papers:** [Shiu et al. 2024](https://doi.org/10.1038/s41586-024-07763-9),
  [Schlegel et al. 2024](https://doi.org/10.1038/s41586-024-07686-5), and
  [Dorkenwald et al. 2024](https://doi.org/10.1038/s41586-024-07558-y)

### Neuron model

- Zhang X, Yang P, Feng J, et al. "Network Structure Governs Drosophila
  Brain Functionality." [arXiv:2404.17128](https://arxiv.org/abs/2404.17128)
  (2024).

## Requirements

- **CUDA path (C++):** NVIDIA CUDA Toolkit 12.x
- **Data conversion:** NumPy, pandas, and PyArrow
- **Python path:** Python 3.10+, [CuPy](https://cupy.dev/) (`pip install cupy-cuda12x`)
- **Web visualizer:** FastAPI, uvicorn (`pip install fastapi uvicorn[standard]`)

## Quick start

### 1. Download and convert the complete CNS

```bash
pip install pandas pyarrow numpy
python download_male_cns.py
```

The official bulk download is approximately 1.1 GB. It produces:

- `male_cns_v1.bin`
- `male_cns_v1.annotations.npz`
- `male_cns_v1.manifest.json`

Use `python download_male_cns.py --annotations-only` to inspect the population
catalog without downloading the connectivity table.

For the legacy FlyWire brain-only dataset, run both
`python download_connectome.py` and `python download_metadata.py`.

### 2a. Run the CUDA simulator

```bash
build.bat
flywire_sim.exe --data male_cns_v1.bin
```

### 2b. Run the Python/CuPy simulator

```bash
pip install cupy-cuda12x
python flywire_sim.py --data male_cns_v1.bin
```

### 3. Web visualizer

```bash
pip install fastapi uvicorn[standard]
python app_server.py --data male_cns_v1.bin
# Open http://127.0.0.1:8000
```

`SimEngine` automatically loads the `.manifest.json` and `.annotations.npz`
companions beside the selected binary and rejects mismatched neuron counts or
dataset identities.

## Training harness

`training_harness.py` schedules annotated upstream populations over simulation
time and scores downstream firing rates against target behaviors. The result is
a normalized reward in `[0, 1]` for an external optimizer or plasticity rule.

```python
from sim_engine import SimEngine
from training_harness import (
    FlyTrainingHarness, RewardRule, StimulusEvent, TrainingScenario,
)

engine = SimEngine("male_cns_v1.bin")
harness = FlyTrainingHarness(engine)
scenario = TrainingScenario(
    name="odor-driven feeding",
    duration=150,
    stimuli=(StimulusEvent("antenna_olf_left", 0, 100, amplitude=0.5),),
    rewards=(
        RewardRule("motor_proboscis", 50, 150, target_rate=0.2),
        RewardRule("motor_pharynx", 50, 150, target_rate=0.1),
    ),
)
result = harness.run_episode(scenario, seed=42)
print(result.reward)
```

Run `python training_harness.py` to print the complete upstream/downstream
population catalog, including exact MaleCNS selectors and fidelity notes. The
catalog includes direct left/right sensory populations for eyes, antennae,
proboscis, all three leg pairs, wings, halteres, notum, and abdomen. Direct
motor readouts cover the proboscis, pharynx, antennae, legs, wings, halteres,
neck, and abdomen, alongside brain-to-VNC descending command populations.

## Project structure

| File | Description |
|---|---|
| `flywire_sim.cu` | Standalone CUDA C++ simulator |
| `flywire_sim.py` | Python/CuPy simulator (runtime-compiled CUDA kernels, no Visual Studio needed) |
| `sim_engine.py` | Simulation engine used by the web server |
| `training_harness.py` | Temporal stimulus scheduling, downstream reward scoring, and neuron population catalog |
| `app_server.py` | FastAPI web server with WebSocket-based live visualizer |
| `download_connectome.py` | Downloads FlyWire v783 data and converts to binary format |
| `download_male_cns.py` | Downloads and converts the seamless MaleCNS brain + ventral nerve cord |
| `download_metadata.py` | Downloads neuron annotation metadata |
| `build.bat` | Build script for the C++ simulator (targets RTX 3080 Ti / SM 8.6) |

## License

MIT
