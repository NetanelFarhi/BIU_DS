# REPORT.md - Telco Customer Churn Classification

**Student:** Netanel Farhi  
**Task:** Option B - Telco Customer Churn Classification  
**Dataset:** Telco Customer Churn, Kaggle  

---

## 1. Business framing

The business problem is customer churn prediction.  
The company wants to identify customers who are likely to leave, so the retention team can focus on the right customers before it is too late.

**Target:** `Churn = Yes`  
**Positive class:** churn customer  
**Model-selection metric:** ROC-AUC  
**Operating metric:** recall / precision at the selected threshold

I selected the model by ROC-AUC because it measures how well the model ranks churn risk across thresholds.  
After choosing the model, I selected the threshold with cross-validation on the training set, based on the tradeoff between missing churn customers and contacting extra customers.

---

## 2. Data preparation

I started with data type checks, missing values, duplicate rows, duplicate customer IDs, unique values, and target distribution.  
`TotalCharges` was stored as text, so I converted it to numeric. This created 11 missing rows, which I removed because the amount was very small.  
I kept `SeniorCitizen` as a binary 0/1 feature because it was already a valid indicator.

---

## 3. EDA and data validation

I checked impossible values such as negative tenure, negative monthly charges, negative total charges, and invalid `SeniorCitizen` values.  
The EDA showed that churn is higher for newer customers, month-to-month contracts, and customers with higher monthly charges.

---

## 4. Results table

| model                     |   accuracy |   precision |   recall |    f1 |   roc_auc |   pr_auc |   threshold |
|:--------------------------|-----------:|------------:|---------:|------:|----------:|---------:|------------:|
| Gradient Boosting         |      0.795 |       0.644 |    0.513 | 0.571 |     0.84  |    0.655 |         0.5 |
| Tuned Gradient Boosting   |      0.796 |       0.645 |    0.519 | 0.575 |     0.839 |    0.654 |         0.5 |
| Logistic Regression       |      0.805 |       0.65  |    0.575 | 0.61  |     0.833 |    0.617 |         0.5 |
| Tuned Random Forest       |      0.794 |       0.647 |    0.495 | 0.561 |     0.833 |    0.631 |         0.5 |
| Tuned Logistic Regression |      0.803 |       0.647 |    0.572 | 0.607 |     0.833 |    0.614 |         0.5 |
| Random Forest             |      0.793 |       0.641 |    0.505 | 0.565 |     0.831 |    0.634 |         0.5 |
| Dummy baseline            |      0.734 |       0     |    0     | 0     |     0.5   |    0.266 |       nan   |

**Selected model:** Logistic Regression  

I selected by ROC-AUC. The models were close, so I preferred the simpler Logistic Regression when the gap was small.  
The selected model achieved ROC-AUC of 0.833 on the test set, compared with the dummy baseline ROC-AUC of 0.500.

The real models are close to each other, around the same ROC-AUC range. Because the score is already stable, I kept the tuning limited and focused on a clear threshold decision instead of adding many more models.

---

## 4.1 Train-test gap

The selected model train-test ROC-AUC gap was 0.015.  
For the tree-based models, I checked the train-test gap because these models can overfit more easily.  
Random Forest used `min_samples_leaf=5`. Gradient Boosting used smaller trees, a lower learning rate, subsampling, and more estimators.


## 4.2 Hyperparameter tuning

I tuned Logistic Regression with GridSearchCV.  
I also used a small RandomizedSearchCV for Random Forest and Gradient Boosting.  
The goal was to check a few reasonable settings without making the search unnecessarily large.

| model                     |   best_cv_roc_auc | best_params                                                                                                                              |
|:--------------------------|------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------|
| Tuned Gradient Boosting   |            0.8493 | {'model__subsample': 0.8, 'model__n_estimators': 400, 'model__min_samples_leaf': 1, 'model__max_depth': 2, 'model__learning_rate': 0.03} |
| Tuned Random Forest       |            0.8467 | {'model__n_estimators': 300, 'model__min_samples_leaf': 10, 'model__max_features': 'sqrt', 'model__max_depth': None}                     |
| Tuned Logistic Regression |            0.8447 | {'model__C': 10.0, 'model__penalty': 'l1', 'model__solver': 'liblinear'}                                                                 |

## 4.3 Feature selection

To avoid choosing an arbitrary number of features, I ranked the original columns by permutation importance on the training set, then compared Top 4, Top 6, Top 8, Top 10, and all features using cross-validation.

| feature_set   |   n_features | features                                                                                                                                                                                                                                                             |   cv_roc_auc_mean |   cv_roc_auc_std |
|:--------------|-------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------:|-----------------:|
| All features  |           19 | gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges |            0.8447 |           0.0055 |
| Top 10        |           10 | tenure, InternetService, Contract, MonthlyCharges, StreamingTV, StreamingMovies, OnlineSecurity, MultipleLines, TechSupport, PaymentMethod                                                                                                                           |            0.8447 |           0.0047 |
| Top 8         |            8 | tenure, InternetService, Contract, MonthlyCharges, StreamingTV, StreamingMovies, OnlineSecurity, MultipleLines                                                                                                                                                       |            0.8429 |           0.0037 |
| Top 6         |            6 | tenure, InternetService, Contract, MonthlyCharges, StreamingTV, StreamingMovies                                                                                                                                                                                      |            0.8393 |           0.0044 |
| Top 4         |            4 | tenure, InternetService, Contract, MonthlyCharges                                                                                                                                                                                                                    |            0.8343 |           0.0055 |

I selected `Top 8` because it gave a simpler model while staying close to the best CV ROC-AUC.  
The selected compact features were: tenure, InternetService, Contract, MonthlyCharges, StreamingTV, StreamingMovies, OnlineSecurity, MultipleLines.

At the selected threshold, the compact model achieved recall 0.711, precision 0.540, F1 0.614, and ROC-AUC 0.830.  
The full selected model achieved recall 0.711, precision 0.553, F1 0.622, and ROC-AUC 0.833.

This comparison shows the tradeoff between a simpler model and the small loss in performance.

## 5. Threshold decision

I selected the operating threshold using cross-validation on the training set, not by searching directly on the test set.  
Selected by cross-validation: best F1 among thresholds with precision >= 0.55 and recall >= 0.60.

On the test set, threshold 0.35 flags 34.2% of customers.  
At this threshold, recall is 0.711 and precision is 0.553.  
The approximate 95% band around recall is ±0.046, based on 374 positive test cases.

---

## 6. Guiding questions

### 1. Accuracy trap

Accuracy is misleading here because the dummy baseline already reaches accuracy of 0.734 by predicting that nobody churns.  
The selected model accuracy at the selected threshold is 0.770, which can be lower than the dummy, but it catches real churn customers.  
That is the accuracy trap: higher accuracy does not mean the model solves the business problem.

### 2. Cost of mistakes

A false positive means the company may contact or offer a retention action to a customer who would have stayed.  
A false negative means the company misses a customer who is actually about to leave.  
I used a cross-validation threshold table on the training set to choose the operating point: lower thresholds catch more churn customers, while higher thresholds reduce unnecessary contacts.

### 3. Is it worth deploying?

The model lifts ROC-AUC from 0.50 for the dummy baseline to about 0.83.  
The models were close, so I would not choose a more complex model unless it gives a clear business improvement.  
I would use the selected model as a prioritization tool, not as an automatic decision engine.

### 4. What drives the model?

The strongest features by permutation importance were tenure, InternetService, Contract.  
These features make business sense because churn is strongly connected to contract type, tenure, service usage, and billing behavior.  
I also compared compact feature sets with cross-validation to check whether fewer columns could keep similar performance.

### 5. Worst mistakes

The worst mistakes are the cases where the model was confident but wrong.  
A typical pattern is long-tenure customers on one-year or two-year contracts who churned anyway, and some high-risk month-to-month customers who stayed.  
These are hard cases because their behavior goes against the main pattern in the data.

### 6. Stability

The selected model ROC-AUC across CV folds was 0.845 ± 0.006.  
I would report the mean and standard deviation together, not a single number.  
This is more honest for stakeholders and avoids over-promising.

### 7. Leakage and split

For this dataset, there is no date column, so I used a stratified train-test split.  
All preprocessing was inside sklearn pipelines, so the test data did not influence imputation, scaling, encoding, or model fitting.  
I removed `customerID` and the raw target column before modeling.

### 8. Production mindset

If this model went live, I would monitor ROC-AUC, recall, precision, false negatives, churn rate, calibration, and feature drift.  
I would retrain the model if the customer mix, pricing, contracts, or churn rate changed materially.  
The model should support retention prioritization, not make customer decisions alone.

---

## 7. Model Card

### Overview
- Task: Predict customer churn.
- Dataset: Telco Customer Churn.
- Target: `Churn = Yes`.
- Positive class: churn.

### Metric and performance
- Model-selection metric: ROC-AUC.
- Operating threshold: 0.35, selected using cross-validation on the training set.
- Selected model: Logistic Regression.
- Test ROC-AUC: 0.833.
- Test recall at selected threshold: 0.711.
- Test precision at selected threshold: 0.553.

### What the model relies on
- Top features: tenure, InternetService, Contract, MonthlyCharges, StreamingTV.
- Compact feature-set selected by CV: Top 8 — tenure, InternetService, Contract, MonthlyCharges, StreamingTV, StreamingMovies, OnlineSecurity, MultipleLines.
- Leakage check: removed `customerID`, target not included in features, preprocessing inside pipeline.
- Multicollinearity check: numeric correlation and VIF-like check on training data only.

### Limitations and failure modes
- The dataset is historical and may not represent future churn behavior.
- The model may over-prioritize some customer groups if the business process changes.
- Probability calibration should be monitored if the model is deployed.

### Real world use
The model should be used as a decision-support tool.  
Recommended use: rank customers by churn risk and let the retention team review the list.

### Monitoring
Monitor ROC-AUC, recall, precision, false negatives, churn rate, probability calibration, feature drift, and offer acceptance rate.  
Retrain if performance drops or if the business process changes.
