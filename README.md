# DuPont
DuPont's formulations problem by Soham Jariwala, Cesar-Claros Olivares and Zijie Wu (ranked in alphabetical order), under the mentoring of Dr. Nick Iovanac at Dupont.

## Part 1. Forward model training
Train the model that predicts properties (water absorptivity, hardness) based on the compositions of (aliased components)

### model_selection.py
Select the best-performing model out of a range of regression models using a nested cross-validation. Gaussian Process Regression (GPR) and K-nearest neighbors (KNN) are selected for its performance being the most superior.

### model_training.py
Optimize the hyperparameters of selected models with grid search, and then train the selected models for prediction.

## Part 2. Reverse model: search for polymer formulation composition that are likely to produce certain property values.
### pso.py
Search for formulation composition that can satisfy certain water absorptivity and/or hardness values. An extra regularizing parameter ("lambda") is introduced for balancing the exactness of property value match vs. confidence in prediction.

**See report.pdf for details.**
