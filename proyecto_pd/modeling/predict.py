"""Inferencia de SalePrice a partir de datos crudos.

Uso desde la raíz del proyecto:
    python -m proyecto_pd.modeling.predict
    python -m proyecto_pd.modeling.predict --input data/raw/test.csv --output data/processed/predictions.csv
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class InferencePipeline:
    """Flujo completo de inferencia: datos crudos -> SalePrice en dólares.

    Selecciona las columnas del modelo, aplica el pipeline ajustado
    (ingeniería de características + modelo, que predice en escala log1p)
    y revierte la transformación del target con expm1.
    """

    def __init__(self, model_pipeline, features, id_column="Id", target="SalePrice"):
        self.model_pipeline = model_pipeline
        self.features = features
        self.id_column = id_column
        self.target = target

    def predict(self, X):
        missing = set(self.features) - set(X.columns)
        if missing:
            raise ValueError(f"Faltan columnas requeridas por el modelo: {sorted(missing)}")
        return np.expm1(self.model_pipeline.predict(X[self.features]))

    def predict_frame(self, X):
        """Predicciones en formato Kaggle: Id y SalePrice."""
        return pd.DataFrame({self.id_column: X[self.id_column], self.target: self.predict(X)})


def main():
    parser = argparse.ArgumentParser(description="Genera predicciones de SalePrice a partir de datos crudos.")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "raw" / "test.csv")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "inference_pipeline.joblib")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "processed" / "predictions.csv")
    args = parser.parse_args()

    inference_pipeline = joblib.load(args.model)
    predictions = inference_pipeline.predict_frame(pd.read_csv(args.input))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(f"{len(predictions)} predicciones guardadas en {args.output}")


if __name__ == "__main__":
    main()
