# Proyecto de Clasificación de Texto con TF-IDF y Naive Bayes

Este proyecto implementa un flujo completo de clasificación de texto, desde la limpieza del texto hasta la optimización de hiperparámetros con **Optuna**. Se utiliza **TF-IDF** para la vectorización del texto y **Naive Bayes** como clasificador principal.

## Estructura del Proyecto

- `load_data.py`
  Carga del dataset.

- `preprocesamiento.py`  
  Limpieza del texto:
  - Conversión a minúsculas.
  - Eliminación de puntuación.
  - Eliminación de espacios y otros caracteres no deseados.

- `tf_idf.py`  
  Construcción de representaciones **TF-IDF** con distintas configuraciones de parámetros (n-gramas, `min_df`, `max_df`, normalización, etc.).

- `objetive_optuna.py`  
  Contiene la función objetivo para Optuna:
  - Preprocesamiento del texto.
  - Vectorización TF-IDF.
  - Entrenamiento de Naive Bayes.
  - Optimización de hiperparámetros.

## Requisitos

```bash
pip install pandas matplotlib numpy seaborn scikit-learn optuna
