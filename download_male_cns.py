"""Download and convert the seamless MaleCNS brain + ventral nerve cord.

Source: https://male-cns.janelia.org/download/
Dataset: male-cns:v1.0 (CC-BY), annotation status ``Traced``.

The official aggregate connectivity contains unsigned contact counts. This
converter assigns a presynaptic sign from the consensus neurotransmitter:
acetylcholine is positive; GABA, glutamate, and histamine are negative.
Neuromodulators and unresolved transmitters are omitted rather than assigned a
made-up fast synaptic polarity. This is a receptor-independent simulation
approximation, not a claim about every postsynaptic response.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import time
import urllib.request

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None


DATASET_ID = "male-cns:v1.0"
BASE_URL = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/"
    "connectome-data/flat-connectome"
)
ARTIFACTS = {
    "annotations": (
        "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        14_483_314,
    ),
    "neurotransmitters": (
        "body-neurotransmitters-male-cns-v1.0.feather",
        43_282_834,
    ),
    "connectivity": (
        "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        1_051_241_946,
    ),
}
SIGN_BY_TRANSMITTER = {
    "acetylcholine": 1.0,
    "gaba": -1.0,
    "glutamate": -1.0,
    "histamine": -1.0,
}


def download_file(url: str, destination: Path, expected_bytes: int) -> None:
    if destination.exists():
        observed = destination.stat().st_size
        if observed != expected_bytes:
            raise RuntimeError(
                f"existing {destination} has {observed} bytes; "
                f"expected {expected_bytes}"
            )
        print(f"  Using cached {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    def progress(blocks: int, block_size: int, total: int) -> None:
        downloaded = min(blocks * block_size, total) if total > 0 else blocks * block_size
        print(
            f"\r  {destination.name}: {downloaded / 1e6:.1f} MB",
            end="",
            flush=True,
        )

    urllib.request.urlretrieve(url, partial, reporthook=progress)
    print()
    observed = partial.stat().st_size
    if observed != expected_bytes:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"downloaded {observed} bytes for {destination.name}; "
            f"expected {expected_bytes}"
        )
    os.replace(partial, destination)


def _require_pandas() -> None:
    if pd is None:
        raise ImportError(
            "MaleCNS conversion requires pandas and pyarrow; "
            "install them with: pip install pandas pyarrow"
        )


def load_traced_annotations(path: Path) -> pd.DataFrame:
    _require_pandas()
    frame = pd.read_feather(path)
    required = {
        "bodyId", "status", "superclass", "class", "subclass", "type",
        "rootSide", "somaSide", "somaNeuromere", "entryNerve", "exitNerve",
        "somaLocation",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"MaleCNS annotations are missing columns: {missing}")
    traced = frame.loc[frame["status"].eq("Traced")].copy()
    traced.sort_values("bodyId", inplace=True)
    traced.reset_index(drop=True, inplace=True)
    if traced.empty or traced["bodyId"].duplicated().any():
        raise ValueError("Traced MaleCNS body IDs must be non-empty and unique")
    return traced


def _string_columns(frame: pd.DataFrame) -> pd.DataFrame:
    _require_pandas()
    result = frame.copy()
    for name in (
        "superclass", "class", "subclass", "type", "rootSide", "somaSide",
        "somaNeuromere", "entryNerve", "exitNerve",
    ):
        result[name] = result[name].fillna("").astype(str)
    return result


def build_population_groups(
    annotations: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build non-overlapping motor readouts and anatomically scoped inputs."""
    frame = _string_columns(annotations)
    sc = frame["superclass"]
    cell_class = frame["class"]
    subclass = frame["subclass"]
    root_side = frame["rootSide"]
    soma_side = frame["somaSide"]
    entry = frame["entryNerve"]
    exit_nerve = frame["exitNerve"]

    def indices(mask: pd.Series | np.ndarray) -> np.ndarray:
        return np.flatnonzero(np.asarray(mask, dtype=np.bool_)).astype(np.int32)

    sensory_masks: dict[str, pd.Series] = {}

    def add_sensory_sides(name: str, mask: pd.Series) -> None:
        sensory_masks[f"{name}_left"] = mask & root_side.eq("L")
        sensory_masks[f"{name}_right"] = mask & root_side.eq("R")

    add_sensory_sides("eye", sc.eq("ol_sensory"))
    add_sensory_sides(
        "antenna_olf", cell_class.eq("olfactory") & entry.eq("AN")
    )
    add_sensory_sides(
        "maxillary_olf", cell_class.eq("olfactory") & entry.eq("MxLbN")
    )
    add_sensory_sides(
        "antenna_mech", cell_class.eq("mechanosensory") & entry.eq("AN")
    )
    add_sensory_sides(
        "proboscis_gust", cell_class.eq("gustatory") & entry.eq("MxLbN")
    )
    add_sensory_sides(
        "pharyngeal_gust",
        cell_class.eq("gustatory") & entry.isin(("aPhN", "PhN")),
    )
    add_sensory_sides("thermo", cell_class.eq("thermosensory"))
    add_sensory_sides("hygro", cell_class.eq("hygrosensory"))

    vnc_sensory = sc.eq("vnc_sensory")
    add_sensory_sides("front_leg", vnc_sensory & entry.eq("ProLN"))
    add_sensory_sides("middle_leg", vnc_sensory & entry.eq("MesoLN"))
    add_sensory_sides("hind_leg", vnc_sensory & entry.eq("MetaLN"))
    add_sensory_sides("wing", vnc_sensory & entry.eq("ADMN"))
    add_sensory_sides("haltere", vnc_sensory & entry.eq("DMetaN"))
    add_sensory_sides("notum", vnc_sensory & entry.eq("PDMN"))
    add_sensory_sides(
        "abdomen", vnc_sensory & entry.str.startswith("Ab", na=False)
    )
    sensory = {
        name: values for name, mask in sensory_masks.items()
        if (values := indices(mask)).size
    }

    motor_masks: dict[str, pd.Series] = {
        "motor_proboscis": sc.eq("cb_motor") & exit_nerve.eq("MxLbN"),
        "motor_pharynx": sc.eq("cb_motor") & exit_nerve.eq("PhN"),
        "motor_antenna": sc.eq("cb_motor") & exit_nerve.eq("AN"),
        "motor_eye": sc.isin(("cb_motor", "cb_efferent")) & exit_nerve.eq("ON"),
        "motor_neck": sc.eq("cb_motor") & exit_nerve.eq("CvN"),
    }

    def add_motor_sides(name: str, mask: pd.Series) -> None:
        motor_masks[f"{name}_left"] = mask & soma_side.eq("L")
        motor_masks[f"{name}_right"] = mask & soma_side.eq("R")

    vnc_motor = sc.eq("vnc_motor")
    add_motor_sides("motor_front_leg", vnc_motor & subclass.eq("fl"))
    add_motor_sides("motor_middle_leg", vnc_motor & subclass.eq("ml"))
    add_motor_sides("motor_hind_leg", vnc_motor & subclass.eq("hl"))
    add_motor_sides("motor_wing", vnc_motor & subclass.eq("wm"))
    add_motor_sides("motor_haltere", vnc_motor & subclass.eq("hm"))
    add_motor_sides("motor_abdomen", vnc_motor & subclass.eq("ad"))
    add_motor_sides("motor_vnc_neck", vnc_motor & subclass.eq("nm"))

    descending = sc.eq("descending_neuron")
    motor_masks["descending_left"] = descending & soma_side.eq("L")
    motor_masks["descending_right"] = descending & soma_side.eq("R")
    motor_masks["descending_center"] = descending & soma_side.eq("M")
    motor = {
        name: values for name, mask in motor_masks.items()
        if (values := indices(mask)).size
    }

    groups = {
        name: values
        for name in sorted(sc[sc.ne("")].unique())
        if (values := indices(sc.eq(name))).size
    }
    return sensory, motor, groups


def _normalized_positions(values: pd.Series) -> np.ndarray:
    positions = np.full((len(values), 3), np.nan, dtype=np.float32)
    for index, value in enumerate(values):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        point = np.asarray(value, dtype=np.float32).reshape(-1)
        if point.size == 3:
            positions[index] = point
    valid = np.isfinite(positions).all(axis=1)
    if not valid.any():
        return np.zeros_like(positions)
    low = positions[valid].min(axis=0)
    high = positions[valid].max(axis=0)
    span = np.where(high > low, high - low, 1.0)
    positions[valid] = ((positions[valid] - low) / span) * 2.0 - 1.0
    positions[~valid] = 0.0
    return positions


def transmitter_signs(
    body_ids: np.ndarray, neurotransmitter_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    _require_pandas()
    table = pd.read_feather(
        neurotransmitter_path, columns=["body", "consensus_nt"]
    )
    if table["body"].duplicated().any():
        raise ValueError("MaleCNS neurotransmitter body IDs must be unique")
    labels = (
        table.set_index("body")["consensus_nt"]
        .reindex(body_ids)
        .fillna("unclear")
        .astype(str)
        .str.lower()
        .to_numpy()
    )
    signs = np.zeros(len(body_ids), dtype=np.float32)
    for transmitter, sign in SIGN_BY_TRANSMITTER.items():
        signs[labels == transmitter] = sign
    return labels, signs


def write_annotations(
    annotations: pd.DataFrame,
    neurotransmitter_path: Path,
    destination: Path,
) -> dict[str, int]:
    frame = _string_columns(annotations)
    body_ids = frame["bodyId"].to_numpy(dtype=np.int64)
    nt_labels, nt_signs = transmitter_signs(body_ids, neurotransmitter_path)
    sensory, motor, groups = build_population_groups(frame)
    positions = _normalized_positions(annotations["somaLocation"])
    side = frame["rootSide"].where(frame["rootSide"].ne(""), frame["somaSide"])
    side = side.map({"L": "left", "R": "right", "M": "center"}).fillna("")

    payload: dict[str, np.ndarray] = {
        "dataset_id": np.array(DATASET_ID),
        "root_ids": body_ids,
        "super_class": frame["superclass"].to_numpy(dtype=str),
        "cell_class": frame["class"].to_numpy(dtype=str),
        "cell_sub_class": frame["subclass"].to_numpy(dtype=str),
        "cell_type": frame["type"].to_numpy(dtype=str),
        "side": side.to_numpy(dtype=str),
        "entry_nerve": frame["entryNerve"].to_numpy(dtype=str),
        "exit_nerve": frame["exitNerve"].to_numpy(dtype=str),
        "soma_neuromere": frame["somaNeuromere"].to_numpy(dtype=str),
        "top_nt": nt_labels.astype(str),
        "nt_sign": nt_signs,
        "pos_x": positions[:, 0],
        "pos_y": positions[:, 1],
        "pos_z": positions[:, 2],
        "group_names": np.array(tuple(groups)),
        "body_sensory_names": np.array(tuple(sensory)),
        "body_motor_names": np.array(tuple(motor)),
    }
    for name, values in groups.items():
        payload[f"group_{name}"] = values
    for name, values in sensory.items():
        payload[f"bsens_{name}"] = values
    for name, values in motor.items():
        payload[f"bmotor_{name}"] = values

    stimulus_names = []
    for name, values in sensory.items():
        label = name.replace("_", " ").title()
        payload[f"stim_{label.replace(' ', '_')}"] = values
        stimulus_names.append(label)
    payload["stim_names"] = np.array(stimulus_names)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    os.replace(temporary, destination)
    return {
        "neurons": len(body_ids),
        "sensory_groups": len(sensory),
        "motor_groups": len(motor),
        "heatmap_groups": len(groups),
        "signed_neurons": int(np.count_nonzero(nt_signs)),
        "unsigned_neurons": int(np.count_nonzero(nt_signs == 0)),
    }


def _arrow_columns(path: Path) -> tuple[list[str], object]:
    import pyarrow as pa
    import pyarrow.ipc as ipc

    source = pa.memory_map(str(path), "r")
    reader = ipc.open_file(source)
    return reader.schema.names, source


def load_connectivity_columns(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import pyarrow.feather as feather

    names, source = _arrow_columns(path)
    source.close()

    def resolve(candidates: tuple[str, ...], role: str) -> str:
        for candidate in candidates:
            if candidate in names:
                return candidate
        raise ValueError(f"cannot resolve {role} column from {names}")

    pre_name = resolve(("body_pre", "bodyPre", "source", "pre_id"), "source")
    post_name = resolve(("body_post", "bodyPost", "target", "post_id"), "target")
    count_name = resolve(("weight", "count", "syn_count"), "contact count")
    table = feather.read_table(
        path, columns=[pre_name, post_name, count_name], memory_map=True
    )
    pre = table.column(0).combine_chunks().to_numpy(zero_copy_only=False)
    post = table.column(1).combine_chunks().to_numpy(zero_copy_only=False)
    counts = table.column(2).combine_chunks().to_numpy(zero_copy_only=False)
    if np.any(pre < 0) or np.any(post < 0) or np.any(counts <= 0):
        raise ValueError("connectivity IDs must be nonnegative and counts positive")
    return (
        pre.astype(np.uint64, copy=False),
        post.astype(np.uint64, copy=False),
        counts.astype(np.float32, copy=False),
    )


def build_signed_graph(
    body_ids: np.ndarray,
    neuron_signs: np.ndarray,
    source_ids: np.ndarray,
    target_ids: np.ndarray,
    contact_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Map body IDs to dense indices and omit edges with unresolved sign."""
    body_ids = np.asarray(body_ids, dtype=np.uint64)
    source_positions = np.searchsorted(body_ids, source_ids)
    target_positions = np.searchsorted(body_ids, target_ids)
    source_valid = source_positions < len(body_ids)
    target_valid = target_positions < len(body_ids)
    source_valid[source_valid] &= (
        body_ids[source_positions[source_valid]] == source_ids[source_valid]
    )
    target_valid[target_valid] &= (
        body_ids[target_positions[target_valid]] == target_ids[target_valid]
    )
    retained = source_valid & target_valid
    pre = source_positions[retained].astype(np.uint32)
    post = target_positions[retained].astype(np.uint32)
    counts = contact_counts[retained].astype(np.float32, copy=False)
    signs = neuron_signs[pre]
    signed = signs != 0
    omitted_unsigned = int(np.count_nonzero(~signed))
    pre = pre[signed]
    post = post[signed]
    weights = counts[signed] * signs[signed]
    if not len(pre):
        raise ValueError("no signed edges remain after filtering")

    if np.any(pre[1:] < pre[:-1]):
        order = np.argsort(pre, kind="stable")
        pre = pre[order]
        post = post[order]
        weights = weights[order]
    scale = np.float32(0.03 / np.abs(weights).mean())
    weights = (weights * scale).astype(np.float32)
    return pre, post, weights, omitted_unsigned


def write_binary(
    destination: Path,
    neuron_count: int,
    pre: np.ndarray,
    post: np.ndarray,
    weights: np.ndarray,
) -> None:
    if len(pre) > np.iinfo(np.uint32).max:
        raise ValueError("edge count exceeds binary format capacity")
    counts = np.bincount(pre, minlength=neuron_count).astype(np.uint64)
    offsets64 = np.empty(neuron_count + 1, dtype=np.uint64)
    offsets64[0] = 0
    np.cumsum(counts, out=offsets64[1:])
    if offsets64[-1] != len(pre) or offsets64[-1] > np.iinfo(np.uint32).max:
        raise ValueError("invalid CSR offsets")
    offsets = offsets64.astype(np.uint32)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with temporary.open("wb") as stream:
        stream.write(struct.pack("<IIII", 0x464C5957, 1, neuron_count, len(pre)))
        offsets.tofile(stream)
        post.astype(np.uint32, copy=False).tofile(stream)
        weights.astype(np.float32, copy=False).tofile(stream)
    os.replace(temporary, destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and convert the seamless MaleCNS brain + VNC"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("male_cns_data"))
    parser.add_argument("--output", type=Path, default=Path("male_cns_v1.bin"))
    parser.add_argument("--annotations-output", type=Path)
    parser.add_argument(
        "--annotations-only", action="store_true",
        help="download the two small metadata files and build population groups",
    )
    args = parser.parse_args()
    if args.annotations_output is None:
        args.annotations_output = args.output.with_suffix(".annotations.npz")

    print("MaleCNS v1.0 seamless brain + ventral nerve cord")
    paths = {}
    needed = ("annotations", "neurotransmitters")
    if not args.annotations_only:
        needed += ("connectivity",)
    for artifact in needed:
        filename, expected_bytes = ARTIFACTS[artifact]
        destination = args.data_dir / filename
        download_file(f"{BASE_URL}/{filename}", destination, expected_bytes)
        paths[artifact] = destination

    annotations = load_traced_annotations(paths["annotations"])
    summary = write_annotations(
        annotations, paths["neurotransmitters"], args.annotations_output
    )
    print(f"Wrote {args.annotations_output}: {summary}")
    if args.annotations_only:
        return

    start = time.perf_counter()
    body_ids = annotations["bodyId"].to_numpy(dtype=np.uint64)
    _, neuron_signs = transmitter_signs(body_ids, paths["neurotransmitters"])
    source_ids, target_ids, contact_counts = load_connectivity_columns(
        paths["connectivity"]
    )
    pre, post, weights, omitted_unsigned = build_signed_graph(
        body_ids, neuron_signs, source_ids, target_ids, contact_counts
    )
    write_binary(args.output, len(body_ids), pre, post, weights)

    manifest = {
        "schema_version": "1.0",
        "dataset_id": DATASET_ID,
        "source_page": "https://male-cns.janelia.org/download/",
        "license": "CC-BY-4.0",
        "body_statuses": ["Traced"],
        "neuron_count": len(body_ids),
        "synapse_count": len(pre),
        "annotation_file": os.path.relpath(
            args.annotations_output, args.output.parent),
        "annotation_neuron_count": summary["neurons"],
        "unresolved_sign_policy": "omit-edge",
        "omitted_unsigned_edges": omitted_unsigned,
        "sign_approximation": {
            transmitter: "excitatory" if sign > 0 else "inhibitory"
            for transmitter, sign in SIGN_BY_TRANSMITTER.items()
        },
        "source_sha256": {
            artifact: _sha256(path) for artifact, path in paths.items()
        },
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {args.output}: {len(body_ids)} neurons, {len(pre)} edges "
        f"in {time.perf_counter() - start:.1f}s"
    )
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
