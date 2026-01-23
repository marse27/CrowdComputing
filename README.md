# CrowdComputing

This repository contains a Python script for cleaning, analyzing, and visualizing survey data exported from Qualtrics.

## Usage

1.	Create a virtual environment (optional but recommended).
2.	Install dependencies:
```pip install -r requirements.txt```
3.	Place the raw Qualtrics CSV export in the raw_data/ directory.
4.	Run the analysis script:
```python pipeline_full.py```

## Output
The script will:
- Clean and filter survey responses
- Save the cleaned dataset to cleaned_data/
- Generate plots in plots/full/
- Print summary statistics and analysis results to the console