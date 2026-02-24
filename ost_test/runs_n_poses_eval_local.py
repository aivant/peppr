"""
Local runner for runs-n-poses evaluation (NeoFold predictions).

Mirrors the logic of flows/runs-n-poses-eval-flow.py without Metaflow/Kubernetes
or any moleculearn dependencies.

Usage:
    python scripts/runs_n_poses_eval_local.py \
        --neofold-prediction-paths /path/to/predictions \
        --ref-dir /path/to/ground_truth \
        --system-df-path /path/to/annotation.parquet \
        --output-path ./scores.parquet

    # Multiple paths (comma-separated):
    python scripts/runs_n_poses_eval_local.py \
        --neofold-prediction-paths /path/to/pred1,/path/to/pred2 \
        --ref-dir /path/to/ground_truth \
        --system-df-path /path/to/annotation.parquet
"""

import argparse
import re
from pathlib import Path
import biotite.structure as struc
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
from vantutil.log import setup_logger
from peppr.cli import _load_system
from peppr.common import standardize
from peppr.match import filter_matched, find_optimal_match
from peppr.metric import BiSyRMSD, LDDTPLIScore, LigandRMSD, PocketAlignedLigandRMSD

LOG = setup_logger()

STRUCTURAL_RMSD_THRESHOLD = 2.0
METRICS = [
    LDDTPLIScore(),
    LigandRMSD(),
    BiSyRMSD(threshold=STRUCTURAL_RMSD_THRESHOLD, inclusion_radius=6.0),
    PocketAlignedLigandRMSD(),
]
METRIC_NAMES = [m.name for m in METRICS]


def find_neofold_models(system_dir: Path) -> list[Path]:
    return list(system_dir.glob("model_*.cif"))


def extract_model_num(name: str) -> str:
    match = re.search(r"(model_\d+)|(model_idx_\d+)", name)
    if not match:
        raise ValueError(f"Model number not found in {name}")
    return match.group(0).replace("_idx", "")


def _evaluate_per_ligand(
    ref_polymer: struc.AtomArray,
    pose_structure: struc.AtomArray,
    ligand_dict: dict[str, struc.AtomArray],
    min_sequence_identity: float = 0.95,
) -> dict[str, dict[str, float]]:
    """
    Evaluate each ligand independently using exhaustive chain matching.

    For each ligand, builds a reference from the polymer + that single ligand,
    runs exhaustive matching against the pose, and computes metrics.
    Exhaustive matching optimizes lDDT-PLI directly when selecting the best
    polymer chain assignment, correctly handling symmetric polymer chains even
    for monoatomic ions.
    """
    results = {}
    for name, lig in ligand_dict.items():
        lig_copy = lig.copy()
        lig_copy.chain_id[:] = name

        # Build per-ligand reference: polymer + this ligand
        ref_single = struc.concatenate([ref_polymer, lig_copy])

        try:
            ref_std = standardize(ref_single, remove_monoatomic_ions=False)
            pose_std = standardize(pose_structure, remove_monoatomic_ions=False)
        except Exception as e:
            LOG.warning(f"Standardize failed for {name}: {e}")
            results[name] = {m.name: np.nan for m in METRICS}
            continue

        try:
            matched_ref, matched_pose = find_optimal_match(
                ref_std,
                pose_std,
                min_sequence_identity=min_sequence_identity,
                use_heuristic=False,
                allow_unmatched_entities=True,
                use_structure_match=True,
            )
        except Exception as e:
            LOG.warning(f"Chain matching failed for {name}: {e}")
            results[name] = {m.name: np.nan for m in METRICS}
            continue

        try:
            ref_filt, pose_filt = filter_matched(matched_ref, matched_pose)
        except Exception as e:
            LOG.warning(f"filter_matched failed for {name}: {e}")
            results[name] = {m.name: np.nan for m in METRICS}
            continue

        # Check that this ligand is present after matching
        lig_mask = ref_filt.hetero
        if not lig_mask.any():
            LOG.warning(f"Ligand {name} not found in matched reference")
            results[name] = {m.name: np.nan for m in METRICS}
            continue

        scores = {}
        for metric in METRICS:
            try:
                scores[metric.name] = metric.evaluate(ref_filt, pose_filt)
            except Exception as e:
                LOG.warning(f"Metric {metric.name} failed for ligand {name}: {e}")
                scores[metric.name] = np.nan
        results[name] = scores

    return results


def evaluate(
    reference: Path,
    model: Path,
    ligands: list[Path] | None = None,
    system_name: str | None = None,
) -> list[tuple[str, dict[str, float]]]:
    """
    Evaluate all ligands in a system against the model pose.

    Uses per-ligand exhaustive chain matching to ensure optimal polymer chain
    assignment for each ligand independently. This correctly handles monoatomic
    ions and symmetric polymer chains by optimizing lDDT-PLI during matching.

    Returns a list of (system_id, scores_dict) tuples, one per ligand.
    """
    ref_structure = _load_system(reference)
    pose_structure = _load_system(model)

    if ligands is None:
        ligand_structure_list = {
            chain_array.chain_id[0]: chain_array
            for chain_array in struc.chain_iter(ref_structure[ref_structure.hetero])
        }
        ref_polymer = ref_structure[~ref_structure.hetero]
    else:
        # Strip any hetero atoms from the receptor (e.g. cofactors) so they
        # don't interfere with matching when the ligand is provided via SDF.
        ref_polymer = ref_structure[~ref_structure.hetero]
        ligand_structure_list = {
            ligand.stem: _load_system(ligand) for ligand in sorted(ligands)
        }

    per_ligand_scores = _evaluate_per_ligand(
        ref_polymer=ref_polymer,
        pose_structure=pose_structure,
        ligand_dict=ligand_structure_list,
    )

    results = []
    for ligand_name, scores in per_ligand_scores.items():
        system_id = f"{system_name}::{ligand_name}" if system_name else ligand_name
        results.append((system_id, scores))
    return results


def run_local_eval(
    prediction_path: Path,
    ref_dir: Path,
    system_df_path: str | None,
) -> pd.DataFrame:
    if system_df_path is not None:
        rnp_system_names = set(pl.read_parquet(system_df_path)["system_name"])
        LOG.info(f"Evaluating against {len(rnp_system_names)} systems from {system_df_path}")
    else:
        rnp_system_names = None
        LOG.info("No system filter applied; will process all systems with a matching ref dir")

    all_rows: list[dict] = []

    for model_system_dir in tqdm(list(prediction_path.glob("*")), desc="Evaluating systems", unit="system"):
        system_name = model_system_dir.name
        ref_path = ref_dir / system_name
        if (rnp_system_names is not None and system_name not in rnp_system_names) or not ref_path.is_dir():
            continue

        LOG.info(f"Comparing {system_name}...")
        ref_cif = ref_path / "receptor.cif"
        ligands = list(ref_path.glob("ligand_files/*.sdf"))

        for model_cif in find_neofold_models(model_system_dir):
            model_num = extract_model_num(model_cif.as_posix())
            system_prefix = system_name + "::" + model_num
            try:
                rows = evaluate(
                    reference=ref_cif,
                    model=model_cif,
                    ligands=ligands,
                    system_name=system_prefix,
                )
            except Exception as e:
                LOG.warning(f"evaluate() failed for {system_prefix}: {e}")
                continue
            for system_id, scores in rows:
                all_rows.append({"system_id": system_id, **scores})

    if all_rows:
        result_df = pd.DataFrame(all_rows, columns=["system_id"] + METRIC_NAMES)
    else:
        result_df = pd.DataFrame(columns=["system_id"] + METRIC_NAMES)

    LOG.info(f"Collected scores for {result_df.shape[0]} ligand poses")
    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Local runs-n-poses evaluation for NeoFold predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--neofold-prediction-paths",
        required=True,
        help="Comma-separated local paths to NeoFold prediction directories",
    )
    parser.add_argument(
        "--ref-dir",
        required=True,
        help="Local path to the ground truth reference directory",
    )
    parser.add_argument(
        "--system-df-path",
        required=False,
        default=None,
        help="Path to the system annotation parquet (local or GCS). "
             "If omitted, all systems with a matching ref dir are processed.",
    )
    parser.add_argument(
        "--output-path",
        default="./scores.parquet",
        help="Local path to write the combined output parquet (default: ./scores.parquet)",
    )
    args = parser.parse_args()

    neofold_paths = [Path(p.strip()) for p in args.neofold_prediction_paths.split(",") if p.strip()]
    ref_dir = Path(args.ref_dir)
    LOG.info(f"Running evaluation for {len(neofold_paths)} NeoFold prediction path(s)")

    all_results = []
    for path in neofold_paths:
        LOG.info(f"--- Evaluating: {path}")
        result_df = run_local_eval(
            prediction_path=path,
            ref_dir=ref_dir,
            system_df_path=args.system_df_path,
        )
        result_df["input_path"] = str(path)
        all_results.append(result_df)

    combined = pd.concat(all_results, ignore_index=True)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    LOG.info(f"Written {combined.shape[0]} pose scores to {output_path}")


if __name__ == "__main__":
    main()
