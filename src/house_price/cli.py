from __future__ import annotations

import argparse
import json
from pathlib import Path

from house_price.data import load_prediction_data, load_training_data
from house_price.model import load_model, save_artifacts, train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="house-price",
        description="Train and use a decision-tree Boston house-price model.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="train and evaluate a model")
    train.add_argument("--data", type=Path, default=Path("boston_housing_data.csv"))
    train.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    train.add_argument("--test-size", type=float, default=0.2)
    train.add_argument("--random-state", type=int, default=42)
    train.add_argument("--cv-folds", type=int, default=5)

    predict = commands.add_parser("predict", help="predict prices for rows in a CSV file")
    predict.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path, default=Path("predictions.csv"))
    return parser


def _train(args: argparse.Namespace) -> int:
    features, target = load_training_data(args.data)
    result = train_model(
        features,
        target,
        test_size=args.test_size,
        random_state=args.random_state,
        cv_folds=args.cv_folds,
    )
    paths = save_artifacts(result, features.columns.tolist(), args.output_dir)
    print(json.dumps(result.metrics, ensure_ascii=False, indent=2))
    print(f"Artifacts saved to {paths['model'].parent.resolve()}")
    return 0


def _predict(args: argparse.Namespace) -> int:
    features = load_prediction_data(args.input)
    model = load_model(args.model)
    output = features.copy()
    output["PREDICTED_MEDV"] = model.predict(features)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output)} predictions to {args.output.resolve()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        return _train(args)
    return _predict(args)


if __name__ == "__main__":
    raise SystemExit(main())
