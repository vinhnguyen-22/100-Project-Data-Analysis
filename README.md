# 100 Project Data Analysis

This project is part of a broader initiative to explore and document various approaches to data analysis using real-world datasets. The repository is organized following best practices for reproducible research and modular data science workflows.

## Objectives

- Perform exploratory data analysis (EDA), feature engineering, and modeling across diverse business problems.
- Develop reusable machine learning pipelines and tools for encoding, training, and prediction.
- Document and present results clearly using Jupyter notebooks and `mkdocs`.

## Project Organization

```
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- Documentation using mkdocs.
│
├── models             <- Trained and serialized models, encoders, and model summaries.
│
├── notebooks
│   ├── Breakfast At The Frat
│   │   ├── EDA.ipynb
│   │   ├── Modeling.ipynb
│   │   └── Pricing.ipynb
│   ├── 01-RFM-kmean.ipynb
│   ├── 02-Customer-Analysis-FMCG.ipynb
│   ├── 03-Data-Mapping.ipynb
│   ├── 04-Customer-Churn-Prediction.ipynb
│   └── 05-Chat-With-PDF.ipynb
│
├── references         <- Manuals, data dictionaries, and supporting literature.
│
├── LICENSE
├── Makefile           <- Command automation for data and training.
├── README.md          <- This file.
└── pyproject.toml     <- Configuration for Python packaging and tooling.
```

## Notable Notebooks

- `04-Customer-Churn-Prediction.ipynb`: Predicts customer churn using the Telco dataset.
- `01-RFM-kmean.ipynb`: Customer segmentation using RFM and K-means clustering.
- `05-Chat-With-PDF.ipynb`: Integrates NLP for document Q\&A interaction.

## Setup

Install dependencies using:

```bash
pip install -r requirements.txt
```

## License

MIT – see the LICENSE file for details.
