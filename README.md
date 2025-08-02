# BMT-EpiPred: Bayesian Multitask Learning for Reliable Prediction of Small-Molecule Epigenetic Modulators

This repository provides the implementation for BMT-EpiPred, a novel computational framework for predicting the activity of small-molecule compounds on various epigenetic targets.

## Introduction

The development of drugs targeting epigenetic regulators is a promising area in therapeutics. However, it faces significant hurdles such as limited selectivity, complex biological networks, and scarcity of data. Computational methods, particularly machine learning, can help overcome these challenges.

BMT-EpiPred is a Bayesian multitask learning (MTL) framework designed to address these issues. It learns from multiple epigenetic targets simultaneously, which improves data efficiency and generalization. A key feature of BMT-EpiPred is its ability to quantify the uncertainty of its predictions. This helps in identifying reliable predictions and flagging potentially erroneous ones for novel molecules, which is crucial for reducing risks and saving resources in drug discovery.

## Key Features

*   **Multitask Learning:** Predicts compound activity across 78 epigenetic targets in a single model, leveraging correlations between targets to enhance performance.
*   **Uncertainty Quantification:** Employs a Bayesian approach to provide probabilistic predictions, allowing users to assess the model's confidence.
*   **Out-of-Distribution (OOD) Detection:** Capable of identifying molecules that are chemically different from its training data, preventing overconfident and incorrect predictions on novel compounds.
*   **High Accuracy:** Achieves state-of-the-art performance compared to conventional single-task models and other MTL approaches.
*   **Reliability:** Integrates uncertainty to distinguish between reliable high-confidence predictions and unreliable ones, adding a layer of trust for experimental validation.

## Data & Model Architecture

The model was developed using data from the ChEMBL database. The training dataset includes 19,527 compounds and their activities against 78 epigenetic targets. For OOD detection evaluation, molecules from the QM9 and Davis datasets were used.

<!-- INSERT FIGURE 1 HERE -->
<p align="center"><img src="images/Figure_1.jpg" alt="BMT-EpiPred model architecture" width="800"/></p>
<!-- <p align="center"><b>[Image Placeholder for Figure 1: Data distribution and t-SNE plots]</b></p> -->

BMT-EpiPred uses a shared feature encoder to create a latent representation of a molecule from its chemical structure (using ECFP fingerprints or a graph-based representation). This shared knowledge is then fed into multiple, independent probabilistic "prediction heads," one for each epigenetic target, to predict the activity.

<!-- INSERT FIGURE 2 HERE -->
<p align="center"><img src="images/Figure_2.jpg" alt="BMT-EpiPred model architecture" width="800"/></p>
<!-- <img src="path/to/figure2.png" alt="Model Architecture" width="800"/> -->
<!-- <p align="center"><b>[Image Placeholder for Figure 2: BMT-EpiPred model architecture]</b></p> -->

## Performance Highlights

### Predictive Accuracy

BMT-EpiPred shows superior performance across five key metrics (Accuracy, F1, MCC, Precision, Recall) when compared to a variety of baseline machine learning models.

<!-- INSERT TABLE 1 HERE -->
<p align="center"><img src="images/Figure_5.jpg" alt="Prediction examples for in-distribution and OOD compounds" width="800"/></p>
<!-- You can create a markdown table or insert an image of the table -->
<!-- <p align="center"><b>[Placeholder for Table 1: Comparative results of experimental evaluations]</b></p> -->

### Uncertainty and Confidence

A core strength of the model is its well-calibrated uncertainty. As shown below, the model's performance metrics (like accuracy) improve as we filter for higher-confidence predictions, demonstrating that its uncertainty measure is reliable.

<!-- INSERT FIGURE 3 HERE -->
<p align="center"><img src="images/Figure_3.jpg" alt="Performance metrics under varying confidence scores" width="800"/></p>
<!-- <img src="path/to/figure3.png" alt="Performance vs Confidence" width="800"/> -->
<!-- <p align="center"><b>[Image Placeholder for Figure 3: Performance metrics under varying confidence scores]</b></p> -->

The model's predictions are more closely aligned with actual outcomes compared to other models, as shown by the calibration curves.

<!-- INSERT FIGURE 4 HERE -->
<p align="center"><img src="images/Figure_4.jpg" alt="Prediction examples for in-distribution and OOD compounds" width="800"/></p>
<!-- <img src="path/to/figure4.png" alt="Calibration Curves" width="800"/> -->
<p align="center"><b>[Image Placeholder for Figure 4: Model calibration curves]</b></p>

### Application in Practice

The model can be used to screen new molecules. The example below shows predictions for two compounds. For the "in-distribution" compound (one that is similar to the training data), the model correctly predicts its active targets with high confidence. For the "OOD" compound, the model still makes predictions, but critically, it assigns them a low confidence score, correctly signaling that the results are not trustworthy.

<!-- INSERT FIGURE 5 HERE -->
<p align="center"><img src="images/Figure_5.jpg" alt="Prediction examples for in-distribution and OOD compounds" width="800"/></p>
<!-- <p align="center"><b>[Image Placeholder for Figure 5: Prediction examples for in-distribution and OOD compounds]</b></p> -->

## Requirements and Installation

To set up the environment and run the code, please follow the steps below.

### 1. Environment Setup

It is highly recommended to use a virtual environment to avoid conflicts with other projects. This project is tested with:

*   **Python**: 3.9+
*   **PyTorch**: 1.13.0+
*   **RDKit**: 2022.09.5+

You can create a new environment using Conda:
```bash
# Create a new conda environment
conda create -n bmt-epipred python=3.9

# Activate the environment

conda activate bmt-epipred





