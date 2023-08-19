# Dendrite-Intern-Test

# Machine Learning Pipeline for Classification and Regression For Iris Dataset

## Overview

This repository contains a Python implementation of a machine learning pipeline designed to handle both classification and regression tasks. The pipeline includes data preprocessing, feature reduction, hyperparameter tuning, and model evaluation.

## Problem Statement

The pipeline aims to automate the process of training machine learning models for different types of prediction tasks. It takes a dataset and a configuration JSON file as input and outputs the best model based on the given hyperparameters and evaluation metrics.

## Features

- Data Preprocessing: Handles missing values and categorical encoding.
- Feature Reduction: Supports various methods like correlation with target, tree-based feature importance, and PCA.
- Model Training: Uses GridSearchCV for hyperparameter tuning.
- Model Evaluation: Provides various metrics like accuracy, AUC, F1 score, MAE, MSE, and RMSE.

## Dependencies

- Python 3.x
- scikit-learn
- pandas
- NumPy

- I have also added requirements file have a look on to those 

## How to Run

1. Clone the repository:

    git clone https://github.com/your-username/your-repo-name.git
    ```

2. Navigate to the project directory:

    cd your-repo-name
    ```

3. Install the required packages:

    pip install -r requirements.txt
    ```

4. Run the main script:

    python main_script.py
    ```

## Code Structure

- `WithPipeline.py`: The main script where the pipeline for data preprocessing, feature reduction, model training and 
			evaluation is executed.I have not added the Comments in this code file Because all comment 
			statements are mentioned in Withoutpipeline.py file.Please have a look if any issue arises.
- `WithoutPipeline.py`: Contains all functions for the everything without pipeline. I have coded without pipeline before 
			to ensure my logic works properly and then i will initalize to the pipeline.


## Configuration

The pipeline uses a JSON file for configuration. The JSON file should contain the following EXAMPLE:


Example:

```json
{
  "design_state_data": {
    "algorithms": {
      "RandomForestClassifier": {
        "is_selected": true,
        "hyperparameters": {
          "n_estimators": 100,
          "max_depth": 10
        }
      }
    }
  },
  "feature_reduction": {
    "feature_reduction_method": "PCA",
    "num_of_features_to_keep": 10
  }
}
```
