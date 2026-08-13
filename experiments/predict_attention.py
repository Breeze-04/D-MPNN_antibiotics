"""Predict and rank herbal compounds with the trained Attention D-MPNN.

The script performs inference only. It does not fit, fine-tune, or otherwise
modify the model. Ranking is performed on unique RDKit-canonical, single-fragment
molecules rather than repeated herb-compound association rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
import torch

from chemprop import data, featurizers, models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("experiments/data/herb_compound.csv"),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("experiments/outputs/attention_smoke/best_model.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/outputs/herb_attention_inference"),
    )
    parser.add_argument("--smiles-column", default="Smiles")
    parser.add_argument("--input-encoding", default="cp1252")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonicalize_smiles(smiles: object) -> tuple[str, str | None]:
    """Return an inference status and a canonical SMILES when eligible."""
    if pd.isna(smiles) or not str(smiles).strip():
        return "missing_smiles", None

    text = str(smiles).strip()
    molecule = Chem.MolFromSmiles(text, sanitize=True)
    if molecule is None:
        return "invalid_smiles", None

    canonical = Chem.MolToSmiles(molecule, canonical=True)
    if "." in canonical:
        return "excluded_multifragment", canonical

    return "predicted", canonical


def aggregate_text(series: pd.Series) -> str:
    """Join unique populated metadata values while preserving input order."""
    seen: set[str] = set()
    values: list[str] = []
    for value in series:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            values.append(text)
    return " | ".join(values)


def canonical_smiles_set(path: Path, smiles_column: str) -> set[str]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path, low_memory=False)
    if smiles_column not in frame.columns:
        return set()

    canonical: set[str] = set()
    for smiles in frame[smiles_column].dropna():
        molecule = Chem.MolFromSmiles(str(smiles), sanitize=True)
        if molecule is not None:
            canonical.add(Chem.MolToSmiles(molecule, canonical=True))
    return canonical


def predict(model: torch.nn.Module, smiles: list[str], batch_size: int, num_workers: int) -> np.ndarray:
    datapoints = [data.MoleculeDatapoint.from_smi(value) for value in smiles]
    dataset = data.MoleculeDataset(
        datapoints,
        featurizers.SimpleMoleculeMolGraphFeaturizer(),
    )
    loader = data.build_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    model.eval()
    batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in loader:
            bmg, vertex_descriptors, extra_descriptors, *_ = batch
            batches.append(model(bmg, vertex_descriptors, extra_descriptors).detach().cpu())

    predictions = torch.cat(batches, dim=0).numpy()
    if predictions.ndim == 2:
        if predictions.shape[1] != 1:
            raise ValueError(
                "This inference script expects a single-output regression model; "
                f"received prediction shape {predictions.shape}."
            )
        predictions = predictions[:, 0]
    return predictions.astype(float, copy=False)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if not args.data_path.exists():
        raise FileNotFoundError(f"Inference data not found: {args.data_path}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"Attention model not found: {args.model_path}")

    frame = pd.read_csv(args.data_path, encoding=args.input_encoding, low_memory=False)
    if args.smiles_column not in frame.columns:
        raise ValueError(
            f"CSV is missing SMILES column {args.smiles_column!r}; "
            f"available columns: {list(frame.columns)}"
        )
    if frame.empty:
        raise ValueError("Inference CSV contains no rows")

    frame.insert(0, "source_row", np.arange(len(frame), dtype=int))
    RDLogger.DisableLog("rdApp.error")
    parsed = frame[args.smiles_column].map(canonicalize_smiles)
    frame["inference_status"] = parsed.map(lambda item: item[0])
    frame["canonical_smiles"] = parsed.map(lambda item: item[1])
    RDLogger.EnableLog("rdApp.error")

    eligible = frame.loc[frame["inference_status"].eq("predicted")].copy()
    if eligible.empty:
        raise RuntimeError("No valid single-fragment molecules are available for inference")

    metadata_columns = [
        column
        for column in (
            "Compound Id",
            "Pref Name",
            "CAS",
            "Pubchem ID",
            "Chembl ID",
            "Formula",
            "Herb ID",
        )
        if column in eligible.columns
    ]
    aggregations = {column: aggregate_text for column in metadata_columns}
    unique_molecules = (
        eligible.groupby("canonical_smiles", as_index=False, sort=False)
        .agg(
            source_row_count=("source_row", "size"),
            **{column: (column, function) for column, function in aggregations.items()},
        )
    )
    if "Herb ID" in eligible.columns:
        herb_counts = eligible.groupby("canonical_smiles")["Herb ID"].nunique(dropna=True)
        unique_molecules["herb_count"] = unique_molecules["canonical_smiles"].map(herb_counts)

    model = models.load_model(args.model_path)
    architecture = (
        f"{type(model.message_passing).__name__} -> "
        f"{type(model.agg).__name__} -> {type(model.predictor).__name__}"
    )
    if type(model.agg).__name__ != "AttentiveAggregation":
        raise ValueError(
            "The supplied model does not use AttentiveAggregation; "
            f"loaded architecture: {architecture}"
        )

    pred_log2 = predict(
        model,
        unique_molecules["canonical_smiles"].tolist(),
        args.batch_size,
        args.num_workers,
    )
    if not np.isfinite(pred_log2).all():
        raise RuntimeError("The model produced non-finite log2(MIC) predictions")

    unique_molecules["pred_log2_mic"] = pred_log2
    unique_molecules["pred_mic_ug_ml"] = np.exp2(pred_log2)
    if not np.isfinite(unique_molecules["pred_mic_ug_ml"]).all():
        raise RuntimeError("Converting predictions to MIC produced non-finite values")

    dataset_smiles = canonical_smiles_set(
        Path("experiments/data/chembl_ecoli_MIC_clean_3296.csv"), "canonical_smiles"
    )
    train_smiles: set[str] = set()
    split_path = Path("experiments/outputs/attention_smoke/splits.csv")
    if split_path.exists():
        split_frame = pd.read_csv(split_path, low_memory=False)
        if {"canonical_smiles", "split"}.issubset(split_frame.columns):
            train_smiles = canonical_smiles_set_from_series(
                split_frame.loc[split_frame["split"].eq("train"), "canonical_smiles"]
            )

    unique_molecules["overlap_model_dataset"] = unique_molecules["canonical_smiles"].isin(
        dataset_smiles
    )
    unique_molecules["seen_in_model_train_split"] = unique_molecules[
        "canonical_smiles"
    ].isin(train_smiles)

    unique_molecules = unique_molecules.sort_values(
        ["pred_mic_ug_ml", "canonical_smiles"], ascending=[False, True]
    ).reset_index(drop=True)
    unique_molecules.insert(0, "rank_high_mic", np.arange(1, len(unique_molecules) + 1))
    low_rank = unique_molecules["pred_mic_ug_ml"].rank(method="first", ascending=True)
    unique_molecules.insert(1, "rank_low_mic", low_rank.astype(int))

    prediction_columns = [
        "canonical_smiles",
        "pred_log2_mic",
        "pred_mic_ug_ml",
        "rank_high_mic",
        "rank_low_mic",
        "overlap_model_dataset",
        "seen_in_model_train_split",
    ]
    frame = frame.merge(
        unique_molecules[prediction_columns],
        how="left",
        on="canonical_smiles",
        validate="many_to_one",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows_path = args.output_dir / "herb_compound_predictions.csv"
    ranking_path = args.output_dir / "herb_molecule_ranking.csv"
    top_path = args.output_dir / "top_highest_predicted_mic.csv"
    excluded_path = args.output_dir / "excluded_structures.csv"

    frame.to_csv(all_rows_path, index=False, encoding="utf-8-sig")
    unique_molecules.to_csv(ranking_path, index=False, encoding="utf-8-sig")
    unique_molecules.head(min(args.top_k, len(unique_molecules))).to_csv(
        top_path, index=False, encoding="utf-8-sig"
    )
    frame.loc[~frame["inference_status"].eq("predicted")].to_csv(
        excluded_path, index=False, encoding="utf-8-sig"
    )

    status_counts = {
        str(status): int(count)
        for status, count in frame["inference_status"].value_counts(dropna=False).items()
    }
    summary = {
        "task": "Attention D-MPNN inference and molecule ranking",
        "ranking_definition": {
            "rank_high_mic": "1 = highest predicted MIC (typically weakest predicted antibacterial activity)",
            "rank_low_mic": "1 = lowest predicted MIC (typically strongest predicted antibacterial activity)",
        },
        "input": {
            "data_path": str(args.data_path.resolve()),
            "encoding": args.input_encoding,
            "rows": int(len(frame)),
            "smiles_column": args.smiles_column,
            "status_counts_by_row": status_counts,
        },
        "model": {
            "model_path": str(args.model_path.resolve()),
            "sha256": file_sha256(args.model_path),
            "architecture": architecture,
        },
        "ranking": {
            "unique_eligible_molecules": int(len(unique_molecules)),
            "top_k_file_rows": int(min(args.top_k, len(unique_molecules))),
            "model_dataset_overlaps": int(unique_molecules["overlap_model_dataset"].sum()),
            "model_training_split_overlaps": int(
                unique_molecules["seen_in_model_train_split"].sum()
            ),
        },
        "outputs": {
            "all_rows": str(all_rows_path.resolve()),
            "unique_molecule_ranking": str(ranking_path.resolve()),
            "top_high_mic": str(top_path.resolve()),
            "excluded_structures": str(excluded_path.resolve()),
        },
    }
    (args.output_dir / "inference_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nHighest predicted MIC molecules:")
    display_columns = [
        column
        for column in (
            "rank_high_mic",
            "Compound Id",
            "Pref Name",
            "pred_log2_mic",
            "pred_mic_ug_ml",
            "herb_count",
            "seen_in_model_train_split",
        )
        if column in unique_molecules.columns
    ]
    print(unique_molecules[display_columns].head(min(10, len(unique_molecules))).to_string(index=False))


def canonical_smiles_set_from_series(series: pd.Series) -> set[str]:
    canonical: set[str] = set()
    for smiles in series.dropna():
        molecule = Chem.MolFromSmiles(str(smiles), sanitize=True)
        if molecule is not None:
            canonical.add(Chem.MolToSmiles(molecule, canonical=True))
    return canonical


if __name__ == "__main__":
    main()
