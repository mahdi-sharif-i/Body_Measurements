# Age Prediction from Body Measurements using Linear Regression

## Project Overview

This project focuses on developing and evaluating a linear regression model to predict an individual's age based on a set of key body measurements. The primary objective is to investigate the correlation between these anthropometric data points and age, and subsequently, to build a predictive model capable of estimating age with reasonable accuracy.

The process encompasses several crucial stages:
1.  **Data Acquisition and Preprocessing**: Loading raw data, selecting relevant features, and ensuring data cleanliness.
2.  **Exploratory Data Analysis (EDA)**: Understanding data distributions, identifying patterns, and quantifying relationships between variables, particularly through correlation analysis.
3.  **Model Development**: Training a linear regression model on a prepared dataset.
4.  **Model Evaluation**: Assessing the model's performance using standard statistical metrics and visualizing prediction accuracy and residual behavior.

This README provides a comprehensive guide to understanding, setting up, running, and interpreting the results of this age prediction model.

## Dataset

The analysis utilizes a dataset named `Body_Measurements.csv`. This CSV file is expected to contain various human body measurements. For the purpose of this project, the following specific features are extracted and used:

*   `Age`: The target variable, representing the age of the individual.
*   `ShoulderToWaist`: Measurement from shoulder to waist.
*   `WaistToKnee`: Measurement from waist to knee.
*   `TotalHeight`: The individual's total height.
*   `ShoulderWidth`: The width of the shoulders.
*   `ArmLength`: The length of the arm.
*   `Hips`: The hip circumference.

The integrity and format of these columns are crucial for the successful execution of the script.

## Prerequisites

To run this project, you need to have Python installed (preferably Python 3.7+). The following Python libraries are essential and must be installed in your environment:

*   **pandas**: For data manipulation and analysis.
*   **numpy**: For numerical operations, especially array manipulation.
*   **matplotlib**: For creating static, animated, and interactive visualizations.
*   **scikit-learn**: For machine learning functionalities, including linear regression, data splitting, and scaling.
*   **seaborn**: For statistical data visualization, built on top of matplotlib.

You can install these dependencies using `pip`, the Python package installer, by running the following command in your terminal:

```bash
pip install pandas numpy matplotlib scikit-learn seaborn
