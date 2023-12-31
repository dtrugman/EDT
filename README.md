# Emotional Decision Trees

This repository is part of an academic research conducted in Reichman University.
The goals of the study were twofold:
1. Psychological study: The study deals with the relationship between personality and emotions in situations of decision making uncertainty.
1. Software engineering study: The study deals with a new data classification algorithm for generating a set of effective models

## Prepare environment

This project uses Python3, and requires a small subset of popular libraries for processing and analysis of data.
The project was developed on MacOS. The code should work on a Win/Linux box as well, but small environment adjustments would be required.

To setup the environment automatically on a MacOS machine, just run: `./setup.sh`. This will:
- Create a local virtual environment with all the Python prerequisites
- Install `graphviz` globally on the machine using `brew`

## Data preparation

The raw data for this research is stored as two separate Mendeley datasets:
- [Ellsberg dataset](https://data.mendeley.com/datasets/p4mz2w36zg/draft?a=fa64a6da-a405-41ef-adcd-62a99b6fc362)
- [Allais dataset - TBD](TBD)

Manually download the files from the Mendeley platform and place them in the `data` directory (which you will need to create).

The final result should be a `data` directory with three files:
- `raw_data_ellsberg.csv`
- `demographic_ellsberg.csv`
- `raw_data_allais.csv`
- `demographic_allais.csv`

To preprocess the data into a canonical format that you can then use for machine-learning, please execute the following notebooks:
- `data_processing_ellsberg.ipynb`
- `data_processing_allais.ipynb`

## Data analysis

- [Ellsberg dataset in DAA matrix representation](./ellsberg_daa_matrix.ipynb)
- [Ellsberg DAA personality and statistics analysis](./ellsberg_daa_personality_statistics.ipynb)
- [Ellsberg K-Means clustering based on emotions only](./ellsberg_kmeans_emotion_only.ipynb)
- [Ellsberg K-Means nested clustering based on personality and emotions](./ellsberg_kmeans_personality_and_emotion.ipynb)
- [Ellsberg classical decision tree analysis with a Gini criterion](./ellsberg_decision_tree.ipynb)
- [Ellsberg emotion decision tree analysis with custom criterion](./ellsberg_emotion_decision_tree.ipynb)