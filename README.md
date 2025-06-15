# Age Prediction from Body Measurements

This project aims to predict an individual's age based on various body measurements using a linear regression model. The analysis involves data loading, cleaning, exploratory data analysis (EDA) including correlation analysis, feature scaling, model training, and evaluation.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Analysis Steps](#analysis-steps)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The core idea of this project is to explore the relationship between several body measurements (e.g., shoulder width, arm length, height) and age. A linear regression model is trained to predict age, and its performance is evaluated using standard metrics like Mean Squared Error (MSE) and R-squared (R²). Visualizations are used to understand the data distribution, correlations, and model performance.

## Dataset

The dataset used for this analysis is `Body_Measurements.csv`. It contains various body measurements, including:
- `Age`
- `ShoulderToWaist`
- `WaistToKnee`
- `TotalHeight`
- `ShoulderWidth`
- `ArmLength`
- `Hips`

Only these specific features are utilized for the prediction task.

## Prerequisites

Before running the code, ensure you have the following Python libraries installed:

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `seaborn`

You can install them using pip:

```bash
pip install pandas numpy matplotlib scikit-learn seaborn
