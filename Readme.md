
# Hybrid Table Annotation

This repository contains Jupyter notebooks for our hybrid table annotation approach, which leverages both Language Models (LMs) and Knowledge Bases (KBs) to automatically annotate tables with column types and relationships.

## Description
The folder `Artifacts` contains:

- **LM annotator:** the two `checkpoints` folders containing the fine-tuned LM models.
- **Annotate.py:** the main pipeline script that contains the KB annotator and calls the LM annotator.
- **Hybrid approach example notebook:** an example showing how to apply our hybrid approach on a table.
- **finetuning.ipynb:** notebook showing how we fine-tuned our CTA model.
- **Experiments and results:** includes scripts to compute metrics and an Excel sheet with manual evaluations of CTA annotations.

To reproduce the CPA results, you need to download the SoTab benchmark dataset: [SoTab](https://webdatacommons.org/structureddata/sotab/).

## Requirements
- Python 3.10+
- Jupyter Notebook or JupyterLab
- Install dependencies with:

```bash
pip install -r requirements.txt
