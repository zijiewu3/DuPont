# DuPont
DuPont's formulations problem by Soham Jariwala, Cesar-Claros Oliveira and Zijie Wu (ranked in alphabetical order), under the mentoring of Dr. Nick Iovanac at Dupont.

## Part 1. Forward model training
Train the model that predicts properties (water absorptivity, hardness) based on the compositions of (aliased components)

### model_selection.py
Select the best-performing model out of a range of regression models using a nested cross-validation. Gaussian Process Regression (GPR) was selected for its performance being the most superior.
