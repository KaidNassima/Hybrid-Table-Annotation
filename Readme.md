french below : 
# Hybrid Table Annotation

This repository contains Jupyter notebooks for our hybrid table annotation approach, which leverages both Language Models (LMs) and Knowledge Bases (KBs) to automatically annotate tables with column types and relationships.

## Description
This repo contains:

- **Annotate.py:** the main pipeline script that contains the KB annotator and calls the LM annotator.
- **Hybrid approach example notebook:** an example showing how to apply our hybrid approach on a table.
- **finetuning.ipynb:** notebook showing how we fine-tuned our Column Type Annotation model.
- **LM annotator:** the two `checkpoints` of  the fine-tuned LM models are found here https://doi.org/10.5281/zenodo.17338765. Once you download them, put each in a folder named checkpoint-215500, and checkpoint_roberta.
- **Experiments and results:** includes scripts to compute metrics and an Excel sheet with manual evaluations of CTA annotations.

To reproduce the Column Property Annotation results, you need to download the SoTab benchmark dataset: [SoTab](https://webdatacommons.org/structureddata/sotab/).

## Requirements
- Python 3.10+
- Jupyter Notebook or JupyterLab
- Install dependencies with:

```bash
pip install -r requirements.txt

 # Annotation hybride de tables

Ce dépot contient des notebooks Jupyter pour notre approche d'annotation hybride de tables, qui exploite à la fois des modèles de languages (LM) et des bases de connaissances (KB) pour annoter automatiquement les tables avec les types de colonnes et les relations.

## Description
Ce dépot contient :

- **Annotate.py :** le script principal du pipeline qui contient l'annotateur KB et appelle l'annotateur LM.
- **Exemple de notebook d'approche hybride :** un exemple montrant comment appliquer notre approche hybride à un tableau.
- **finetuning.ipynb :** notebook montrant comment nous avons affiné notre modèle d'extraction de types.
- **Annotateur LM :** les deux « checkpoints» des modèles LM affinés se trouvent ici https://doi.org/10.5281/zenodo.17338765. Une fois que vous les avez téléchargés, placez-les dans un dossier nommé checkpoint-215500 et checkpoint_roberta.
- **Expériences et résultats :** comprend des scripts pour calculer les métriques et une feuille Excel avec des évaluations manuelles des annotations "Types de colonnes".

Pour reproduire les résultats CPA "relations entre colonnes", vous devez télécharger l'ensemble de données de référence SoTab : [SoTab](https://webdatacommons.org/structureddata/sotab/).

## Configuration requise
- Python 3.10+
- Jupyter Notebook ou JupyterLab
- Installez les dépendances avec :

```bash
pip install -r requirements.txt
