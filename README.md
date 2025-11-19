
## Workflow Summary

The notebook follows these steps:

**Data Loading & Cleaning**
- Handles missing values
- Encodes categorical variables (CatBoost/LGB-friendly)
- Standardizes or normalizes numeric features (for NN)

**Exploratory Data Analysis**
- Feature distribution checks
- Outcome imbalance review
- Basic correlations and missingness patterns

**Model Training**
- LightGBM with tuned boosting parameters
- CatBoost with categorical handling
- Neural network built using Keras/PyTorch (depending on your implementation)

**Model Ensembling**
- Weighted averaging
- Optional stacking meta-learner
- Cross-validated performance comparison

**Evaluation**
- ROC-AUC / PR-AUC
- Calibration curves
- Feature importance (for LGB/CatBoost)

## Dependencies

Install required packages:

```bash
pip install numpy pandas scikit-learn lightgbm catboost xgboost tensorflow torch matplotlib
