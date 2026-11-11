# proyecto_pd

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Proyecto del curso Product Development - Maestria BI

## Datos

Los datos no se versionan. Descarga `train.csv`, `test.csv`, `data_description.txt` y `sample_submission.csv` de la competencia [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) de Kaggle y colócalos en `data/raw/`.

## Entorno

```
conda create -n env_house_pricing_a python=3.11
conda activate env_house_pricing_a
pip install -r requirements.txt
```

## Flujo de trabajo

Ejecuta los notebooks en orden. Los modelos (`models/*.joblib`) y los datos procesados no se versionan: se generan al correrlos.

1. `1.0-jec-analisis-exploratorio`: análisis exploratorio y selección de features (genera `pipeline_config.yaml`).
2. `2.0-jec-preprocesamiento-modelo`: pipeline de ingeniería de características.
3. `3.0-jec-entrenamiento-modelos`: torneo de modelos con validación cruzada, reentrenamiento del ganador, exportación de resultados a `reports/model_results.json` y registro en MLflow (champion y challengers).
4. `4.0-jec-inferencia`: predicciones sobre `test.csv` con el champion del Model Registry y pipeline de inferencia.

## MLflow

El notebook 3 registra cada torneo en el experimento `house-pricing-saleprice` (un run padre y un run por modelo) y publica los tres mejores modelos en el Model Registry con los aliases `champion-<modelo>`, `challenger-1-<modelo>` y `challenger-2-<modelo>`. El backend es local (`mlflow.db` y `mlruns/`, fuera del control de versiones). Para explorarlo, en la raíz del proyecto:

```
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

y abre http://127.0.0.1:5000.

Para generar predicciones desde la terminal, después de correr los notebooks 3 y 4:

```
python -m proyecto_pd.modeling.predict --input data/raw/test.csv --output data/processed/predictions.csv
```

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         proyecto_pd and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── proyecto_pd   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes proyecto_pd a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

