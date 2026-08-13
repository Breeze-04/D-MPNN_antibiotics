"""Train a reproducible Chemprop D-MPNN regression model with attention pooling.

Architecture: BondMessagePassing -> AttentiveAggregation -> RegressionFFN (MLP).

This script intentionally mirrors ``experiments/baseline.py``. The experimental
change is limited to replacing MeanAggregation with Chemprop's official
AttentiveAggregation implementation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from lightning import pytorch as pl
import lightning
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import numpy as np
import pandas as pd
import torch

from chemprop import data, featurizers, models, nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--target-columns", nargs="+", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiments/outputs/attention_smoke")
    )
    parser.add_argument("--split-type", default="RANDOM", choices=list(data.SplitType.keys()))
    parser.add_argument("--split-sizes", nargs=3, type=float, default=(0.8, 0.1, 0.1))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--message-hidden-dim", type=int, default=300)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--ffn-hidden-dim", type=int, default=300)
    parser.add_argument("--ffn-num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--accelerator", default="auto", choices=("auto", "cpu", "gpu"))
    return parser.parse_args()


def validate_args(args: argparse.Namespace, frame: pd.DataFrame) -> None:
    required = [args.smiles_column, *args.target_columns]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("CSV contains no rows")
    if frame[args.smiles_column].isna().any():
        raise ValueError("SMILES column contains missing values")
    if frame[args.target_columns].notna().sum().sum() == 0:
        raise ValueError("All target values are missing")
    if any(size <= 0 for size in args.split_sizes):
        raise ValueError("All train/validation/test split sizes must be positive")
    if not math.isclose(sum(args.split_sizes), 1.0, rel_tol=0, abs_tol=1e-8):
        raise ValueError("Train/validation/test split sizes must sum to 1")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, names: list[str]) -> dict:
    per_target: dict[str, dict[str, float | int | None]] = {}
    for index, name in enumerate(names):
        mask = np.isfinite(y_true[:, index]) & np.isfinite(y_pred[:, index])
        actual = y_true[mask, index]
        predicted = y_pred[mask, index]
        if len(actual) == 0:
            per_target[name] = {"n": 0, "rmse": None, "mae": None, "r2": None}
            continue
        residual = actual - predicted
        denominator = np.square(actual - actual.mean()).sum()
        per_target[name] = {
            "n": int(len(actual)),
            "rmse": float(np.sqrt(np.square(residual).mean())),
            "mae": float(np.abs(residual).mean()),
            "r2": float(1 - np.square(residual).sum() / denominator)
            if len(actual) > 1 and denominator > 0
            else None,
        }

    macro = {}
    for metric in ("rmse", "mae", "r2"):
        values = [result[metric] for result in per_target.values() if result[metric] is not None]
        macro[metric] = float(np.mean(values)) if values else None
    return {"macro_average": macro, "per_target": per_target}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(args.seed, workers=True)

    frame = pd.read_csv(args.data_path)
    validate_args(args, frame)
    smiles = frame[args.smiles_column].astype(str).to_numpy()
    targets = frame[args.target_columns].apply(pd.to_numeric, errors="raise").to_numpy(float)
    datapoints = [
        data.MoleculeDatapoint.from_smi(smiles_value, target)
        for smiles_value, target in zip(smiles, targets)
    ]
    if any(datapoint.mol is None for datapoint in datapoints):
        bad_rows = [i for i, datapoint in enumerate(datapoints) if datapoint.mol is None]
        raise ValueError(f"Invalid SMILES at zero-based CSV data rows: {bad_rows[:20]}")

    train_indices, val_indices, test_indices = data.make_split_indices(
        [datapoint.mol for datapoint in datapoints],
        split=args.split_type,
        sizes=tuple(args.split_sizes),
        seed=args.seed,
    )
    train_idx, val_idx, test_idx = train_indices[0], val_indices[0], test_indices[0]
    train_data, val_data, test_data = data.split_data_by_indices(
        datapoints, train_indices, val_indices, test_indices
    )

    split_frame = frame.copy()
    split_frame.insert(0, "source_row", np.arange(len(split_frame)))
    split_frame["split"] = ""
    split_frame.loc[train_idx, "split"] = "train"
    split_frame.loc[val_idx, "split"] = "validation"
    split_frame.loc[test_idx, "split"] = "test"
    split_frame.to_csv(args.output_dir / "splits.csv", index=False)

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    train_dataset = data.MoleculeDataset(train_data[0], featurizer)
    target_scaler = train_dataset.normalize_targets()
    val_dataset = data.MoleculeDataset(val_data[0], featurizer)
    val_dataset.normalize_targets(target_scaler)
    test_dataset = data.MoleculeDataset(test_data[0], featurizer)

    train_loader = data.build_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    val_loader = data.build_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    test_loader = data.build_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    message_passing = nn.BondMessagePassing(
        d_h=args.message_hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
    )

    # The only architectural change relative to baseline.py:
    # learn one scalar attention score per final atom representation and
    # use the within-molecule softmax weights for graph-level pooling.
    aggregation = nn.AttentiveAggregation(output_size=message_passing.output_dim)

    predictor = nn.RegressionFFN(
        n_tasks=len(args.target_columns),
        input_dim=message_passing.output_dim,
        hidden_dim=args.ffn_hidden_dim,
        n_layers=args.ffn_num_layers,
        dropout=args.dropout,
        output_transform=nn.UnscaleTransform.from_standard_scaler(target_scaler),
    )
    model = models.MPNN(
        message_passing,
        aggregation,
        predictor,
        metrics=[nn.RMSE(), nn.MAE(), nn.R2Score()],
    )

    checkpoint = ModelCheckpoint(
        dirpath=args.output_dir / "checkpoints",
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=1,
        max_epochs=args.epochs,
        logger=CSVLogger(save_dir=args.output_dir, name="logs"),
        callbacks=[
            checkpoint,
            EarlyStopping(monitor="val_loss", mode="min", patience=args.patience),
        ],
        deterministic=True,
    )
    trainer.fit(model, train_loader, val_loader)

    best_model = models.MPNN.load_from_checkpoint(checkpoint.best_model_path)
    trainer.test(best_model, dataloaders=test_loader)
    prediction_batches = trainer.predict(best_model, dataloaders=test_loader)
    predictions = torch.cat(prediction_batches, dim=0).detach().cpu().numpy()
    if predictions.ndim == 1:
        predictions = predictions[:, None]

    test_frame = frame.iloc[test_idx].copy()
    test_frame.insert(0, "source_row", test_idx)
    for index, target_name in enumerate(args.target_columns):
        test_frame[f"pred_{target_name}"] = predictions[:, index]
    test_frame.to_csv(args.output_dir / "test_predictions.csv", index=False)

    metrics = regression_metrics(targets[test_idx], predictions, args.target_columns)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    config = vars(args).copy()
    config["data_path"] = str(args.data_path.resolve())
    config["output_dir"] = str(args.output_dir.resolve())
    config["split_sizes"] = list(args.split_sizes)
    config["architecture"] = (
        "BondMessagePassing -> AttentiveAggregation -> RegressionFFN"
    )
    config["attention"] = {
        "implementation": "chemprop.nn.AttentiveAggregation",
        "input_output_dim": message_passing.output_dim,
        "added_trainable_parameters": sum(
            parameter.numel() for parameter in aggregation.parameters()
        ),
    }
    config["best_checkpoint"] = str(Path(checkpoint.best_model_path).resolve())
    config["versions"] = {
        "chemprop": __import__("chemprop").__version__,
        "torch": torch.__version__,
        "lightning": lightning.__version__,
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    models.save_model(args.output_dir / "best_model.pt", best_model, args.target_columns)

    print(f"Best checkpoint: {checkpoint.best_model_path}")
    print(f"Results: {args.output_dir.resolve()}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
