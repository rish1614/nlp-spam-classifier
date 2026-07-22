<div align="center">

# NLP Text Classification Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning architecture for text classification, transitioning from classical Natural Language Processing to modern Transformer-based deep learning. 

[Explore the Code](#project-architecture) · [Report Bug](#) · [Request Feature](#)

</div>

---

## Overview

This project serves as a comprehensive pipeline for classifying SMS messages and emails as Spam or Ham (Safe). It is designed to be highly modular, allowing developers to experiment with various NLP techniques ranging from standard Bag of Words to fine-tuning HuggingFace Transformers. 

**Key Features:**
* **Progressive Complexity:** Includes 7 distinct Jupyter notebooks moving from basic EDA to DistilBERT.
* **Modular Source Code:** Production-ready `src/` directory for text cleaning, feature extraction, and inference.
* **Dual Deployment:** Features both a Streamlit dashboard for rapid prototyping and a Flask application for standard web deployment.
* **Automated Testing:** Integrated unit testing suite using `pytest`.

<br>

<div align="center">
  <!-- NOTE: Replace the src link below with a path to an actual .gif screen recording of your app once you upload it to GitHub -->
  <img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%">
  <br>
  <i>(Tip: Record a short GIF of your Streamlit app working and place it here by changing the image source in the README)</i>
</div>

<br>

---

## Project Architecture

<details>
<summary>Click to expand the full directory tree</summary>

```text
nlp-spam-classifier/
├── app/
│   ├── app.py                 # Streamlit application
│   ├── flask_app.py           # Flask application
│   └── templates/
│       └── index.html         # Tailwind CSS frontend for Flask
├── data/
│   ├── raw/                   # Original dataset
│   └── processed/             # Cleaned and tokenized datasets
├── models/                    # Serialized .pkl models and vectorizers
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb # Text Normalization
│   ├── 03_baseline_model.ipynb# Naive Bayes + BoW
│   ├── 04_tfidf_model.ipynb   # TF-IDF + SVM/Logistic Regression
│   ├── 05_word_embeddings.ipynb # Word2Vec implementation
│   ├── 06_transformer_model.ipynb # DistilBERT fine-tuning
│   └── 07_error_analysis.ipynb# False positive/negative evaluation
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       # Cleaning and lemmatization logic
│   ├── features.py            # Vectorization wrappers
│   ├── train.py               # Model training pipeline
│   ├── evaluate.py            # Metrics and confusion matrices
│   └── predict.py             # Live inference engine
├── tests/
│   └── test_preprocessing.py  # Pytest suite
├── config.yaml                # Global variables and hyperparameters
├── environment.yml            # Conda environment build file
├── requirements.txt           # Pip dependencies
└── README.md
