<div align="center">

# NLP Text Classification Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning architecture for text classification, transitioning from classical Natural Language Processing to modern Transformer-based deep learning. 

[Explore the Code](#project-architecture) · [Dataset Info](#data-provenance) · [Deployment](#deployment-guide)

</div>

---

## Overview

This project serves as a comprehensive, production-ready pipeline for classifying SMS messages and emails as Spam or Ham (Safe). It is designed to be highly modular, functioning both as a robust classification tool and an educational progression from standard statistical models (Bag of Words) to state-of-the-art Deep Learning (HuggingFace Transformers).

**Key Features:**
* **Progressive Complexity:** 7 distinct Jupyter notebooks structured as a learning curriculum.
* **Modular Source Code:** Production-ready `src/` directory for text cleaning, feature extraction, and inference.
* **Dual Deployment:** Features both a Streamlit dashboard for rapid prototyping and a Flask application for traditional web deployment.
* **Automated Testing:** Integrated unit testing suite using `pytest`.

<br>

<div align="center">
  <!-- NOTE: Replace the src link below with a path to an actual .gif screen recording of your app once you upload it to GitHub -->
  <img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%">
  <br>
  <i>(Demo: Streamlit Interface running local inference via Naive Bayes)</i>
</div>

<br>

---

## Project Architecture

<details open>
<summary><b>Click to collapse/expand the directory tree</b></summary>

```text
nlp-spam-classifier/
├── app/
│   ├── app.py                 # Streamlit interactive application
│   ├── flask_app.py           # Flask web server
│   └── templates/
│       └── index.html         # Tailwind CSS frontend for Flask
├── data/
│   ├── raw/                   # Immutable raw UCI dataset (spam.tsv)
│   ├── processed/             # Ephemeral cleaned data (.csv)
│   └── README.md              # Data dictionary and schema rules
├── models/                    # Serialized .pkl models and vectorizers (Git-ignored)
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb # Text Normalization Pipeline
│   ├── 03_baseline_model.ipynb# Naive Bayes + BoW
│   ├── 04_tfidf_model.ipynb   # TF-IDF + SVM/Logistic Regression
│   ├── 05_word_embeddings.ipynb # Word2Vec semantic mapping
│   ├── 06_transformer_model.ipynb # DistilBERT fine-tuning
│   └── 07_error_analysis.ipynb# False positive/negative evaluation
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       # Reusable cleaning & lemmatization logic
│   ├── features.py            # Vectorization wrappers
│   ├── train.py               # Model training execution
│   ├── evaluate.py            # Core metrics and confusion matrices
│   └── predict.py             # Live inference engine for web apps
├── tests/
│   └── test_preprocessing.py  # Pytest suite for data integrity
├── config.yaml                # Global variables and hyperparameters
├── environment.yml            # Conda environment build file
├── requirements.txt           # Pip dependencies fallback
└── README.md
