# Optimizing Industrial Operations Through Predictive Maintenance Modeling
## Predictive Maintenance Using Machine Learning

**Author:** Reeves Gonah  
**Date** May 5, 2026  
**Data Source:** [AI4I 2020 Predictive Maintenance Dataset](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020)  
___
### Overview
This project builds a classification model to predict whether an industrial machine is in a failure‑prone state based on real‑time sensor data. The goal is to shift maintenance operations from reactive (fix‑when‑broken) or rigid scheduled maintenance to a data‑driven predictive maintenance strategy, reducing unplanned downtime and repair costs.

Using the AI4I 2020 Predictive Maintenance Dataset, two supervised machine learning models—Logistic Regression and a Decision Tree Classifier—were developed, compared, optimized, and evaluated. The final model provides actionable insights for maintenance teams, enabling targeted inspections and preventing unexpected breakdowns.
___
### Business Understanding  
**Business Context**  
A mid‑sized manufacturer, Apex Precision Components Ltd., is experiencing rising maintenance costs and frequent unplanned machine downtime across its CNC and automated machining lines. Sensor data (temperature, torque, rotational speed, tool wear) is already collected, offering an opportunity to detect failure‑prone conditions before a breakdown occurs.  

**Business Problem**
The company currently relies on reactive or time‑based maintenance, leading to emergency repairs, production stoppages, and higher costs. A data‑driven method is needed to predict failures in advance so that maintenance can be scheduled proactively.  

**Business Objectives**
Develop a binary classification model that uses sensor readings to predict machine failure risk.  
Use predictions to trigger inspections only when risk is elevated, thereby reducing unplanned outages and controlling maintenance expenses.  

**Success Criteria**
Because missed failures (false negatives) are far more costly than unnecessary inspections (false positives), recall is the primary metric (target ≥ 70–75%). Precision must remain above roughly 50% to ensure alerts are actionable. The F1‑score is used for overall model comparison.
___

### Dataset & Methodology
**<u>Data</u>**
**Source:** AI4I 2020 Predictive Maintenance Dataset (10,000 synthetic observations of a milling machine).

**Features:** Air & Process Temperature, Rotational Speed, Torque, Tool Wear, Machine Type (categorical), and a binary target Machine failure.

**Class Imbalance:** Only ~3.4% of records represent failures, addressed using SMOTE oversampling.

**<u>Preprocessing</u>**
**Dropped:** Non‑predictive columns (UDI, Product ID, failure‑mode indicators).

**Feature Engineering:** Created Temp_Difference (Process – Air) and Mechanical_Power (Torque × Rotational Speed) to better capture mechanical and thermal stress.

**Encoding:** One‑hot encoding for Machine Type.

**Scaling:** Standard scaling applied (important for Logistic Regression).

**Train‑Test Split:** 70% training / 30% test with a fixed random state (73) to ensure reproducibility.

**<u>Modeling Approach</u>**
**Baseline Models:** Unregularized Logistic Regression, and a Decision Tree (default parameters, using entropy for splits). Both trained on the SMOTE‑balanced training set.

**Optimization:**

**Logistic Regression:** tuned regularization strength (C) and the classification threshold to improve precision/recall balance.

**Decision Tree:** hyperparameters (max_depth, min_samples_split, min_samples_leaf) optimized via grid search with 5‑fold cross‑validation.

**Evaluation:** Confusion matrices, accuracy, precision, recall, F1‑score, and 5‑fold cross‑validation.
___

### Key Findings
**Feature Importance**
Rotational Speed, Mechanical Power, and Tool Wear consistently emerged as the strongest predictors of machine failure.  
Temperature variables and Machine Type contributed less, indicating that mechanical load and cumulative wear are the primary failure drivers.

**Model Comparison**
Model	Recall (Test)	F1‑Score (Test)	Precision (Test)	Accuracy (Test)
Baseline Logistic Regression	94.9%	0.13	7.2%	59.7%
Baseline Decision Tree	76.8%	0.61	50.3%	96.7%
Optimized Logistic Regression*	75.8%	0.33	17.3%	86.9%
Optimized Decision Tree	77.8%	0.61	50.7%	96.8%
Threshold set to 0.6 after tuning C and class‑weight balancing.

**Why the Decision Tree outperforms:**
The decision tree naturally captures non‑linear interactions among sensor variables (e.g., high torque and high tool wear and low rotational speed) that are characteristic of mechanical systems. Logistic Regression, even after tuning, struggles to balance recall and precision effectively for this imbalanced problem.
___
### Final Model
The optimized Decision Tree was selected as the final model. It provides:

77.8% recall – catching the majority of impending failures.

50.7% precision – meaning about half of the alerts are true failures, keeping false alarms operationally manageable.

96.8% accuracy – indicating strong overall classification.

Cross‑validation confirmed consistent performance across data folds (mean recall 75%, mean F1 0.77).
___
### Recommendations
Deploy the Decision Tree model as a decision‑support tool that flags high‑risk machines based on current sensor readings (Rotational Speed, Mechanical Power, Tool Wear).  
Set an alert threshold such that maintenance teams are notified when the model predicts a failure‑prone state, allowing scheduled inspections rather than emergency repairs.  
Focus monitoring on machines with high tool wear and abnormal power or speed readings, as these conditions most strongly correlate with impending failures.
Integrate with existing CMMS/SCADA systems to automate data ingestion and alert routing.
___
##### Limitations & Assumptions
The data is synthetic and may not capture all real‑world failure patterns.  
SMOTE was used to balance classes; while it helps training, it may slightly distort the natural failure distribution.  
The model does not consider temporal trends – it predicts failure risk from a single observation rather than time‑based patterns.  
Cross‑validation was performed on the original (pre‑SMOTE) training set to avoid data leakage.
___
##### Future Improvements
Incorporate time‑series features (e.g., rate of change in tool wear) to capture degradation trends.  
Explore ensemble methods (Random Forest, XGBoost) that often handle imbalance better and provide more interpretable feature importance.  
Implement cost‑sensitive learning to directly optimize the cost trade‑off between false alarms and missed failures.  
Validate the model on real‑world data before full production deployment.
___
##### Repository Structure
index.ipynb – Full Jupyter Notebook containing all analysis, modeling, and visualisations.

ai4i2020.csv – The dataset (not included in the repository; available from Kaggle).

Maintenance_Engineering.jpeg – An image used in the notebook.

README.md – This file.

Non-Technical Executive Summary - Non-Technical Presentation Slides in PDF format