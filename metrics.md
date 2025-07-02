# ML Metrics Guide: When to Use What

## Core Classification Metrics

### Accuracy
**Definition**: Proportion of correct predictions out of total predictions  
**Formula**: `(TP + TN) / (TP + TN + FP + FN)`  
**When to use**: Balanced datasets with roughly equal class distribution  
**When NOT to use**: Imbalanced datasets (e.g., 99% negative, 1% positive)

### Precision
**Definition**: Of all positive predictions, how many were actually correct?  
**Formula**: `TP / (TP + FP)`  
**Focus**: Minimizing false positives  
**Use when**: The cost of false positives is high

### Recall (Sensitivity, True Positive Rate)
**Definition**: Of all actual positives, how many did we catch?  
**Formula**: `TP / (TP + FN)`  
**Focus**: Minimizing false negatives  
**Use when**: The cost of missing positive cases is high

### F1 Score
**Definition**: Harmonic mean of precision and recall  
**Formula**: `2 * (Precision * Recall) / (Precision + Recall)`  
**Use when**: You need a balance between precision and recall, especially with imbalanced classes

### AUC-ROC (Area Under the ROC Curve)
**Definition**: Probability that the model ranks a random positive example higher than a random negative example  
**Range**: 0.5 (random) to 1.0 (perfect)  
**Use when**: 
- Comparing models at various thresholds
- Need a threshold-independent metric
- Binary classification with moderate class imbalance

### AUC-PR (Area Under Precision-Recall Curve)
**Definition**: Summary of precision-recall trade-offs at all thresholds  
**Use when**: Severe class imbalance (better than AUC-ROC for rare events)

### Specificity (True Negative Rate)
**Definition**: Of all actual negatives, how many did we correctly identify?  
**Formula**: `TN / (TN + FP)`  
**Use when**: Correctly identifying negatives is important

### Matthews Correlation Coefficient (MCC)
**Definition**: Correlation between predicted and actual classifications  
**Formula**: `(TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))`  
**Range**: -1 to +1 (0 means random prediction)  
**Use when**: Balanced evaluation across all confusion matrix cells

## Regression Metrics

### Mean Absolute Error (MAE)
**Definition**: Average absolute difference between predicted and actual values  
**Formula**: `(1/n) * Σ|y_actual - y_predicted|`  
**Use when**: All errors are equally important, robust to outliers

### Mean Squared Error (MSE)
**Definition**: Average squared difference between predicted and actual values  
**Formula**: `(1/n) * Σ(y_actual - y_predicted)²`  
**Use when**: Large errors are particularly undesirable

### Root Mean Squared Error (RMSE)
**Definition**: Square root of MSE  
**Formula**: `sqrt(MSE)`  
**Use when**: Want error in same units as target variable

### R² (Coefficient of Determination)
**Definition**: Proportion of variance in dependent variable predictable from independent variables  
**Range**: 0 to 1 (can be negative for poor models)  
**Use when**: Comparing models or explaining model performance to stakeholders

### Mean Absolute Percentage Error (MAPE)
**Definition**: Average percentage error  
**Formula**: `(100/n) * Σ|((y_actual - y_predicted) / y_actual)|`  
**Use when**: Error relative to magnitude matters, but beware of values near zero

## Practical Use Case Scenarios

### 1. Medical Diagnosis (e.g., Cancer Detection)
**Primary Metrics**: Recall + Specificity  
**Secondary**: AUC-ROC, F1 Score  
**Why**: Missing a cancer case (false negative) is far worse than additional tests (false positive)

### 2. Email Spam Detection
**Primary Metrics**: Precision + Recall balance (F1 Score)  
**Secondary**: AUC-ROC  
**Why**: Both spam in inbox and legitimate emails in spam are problematic

### 3. Credit Card Fraud Detection
**Primary Metrics**: Recall + AUC-PR  
**Secondary**: Precision at specific recall thresholds  
**Why**: Extreme class imbalance (99.9% legitimate), missing fraud is costly

### 4. Customer Churn Prediction
**Primary Metrics**: F1 Score + AUC-ROC  
**Secondary**: Precision-Recall trade-off analysis  
**Why**: Balance between targeting right customers and not missing at-risk ones

### 5. Manufacturing Quality Control
**Primary Metrics**: Recall + Specificity  
**Secondary**: Overall accuracy (if balanced)  
**Why**: Defective products reaching customers is critical to avoid

### 6. Search/Recommendation Systems
**Primary Metrics**: Precision@K, Recall@K  
**Secondary**: Mean Average Precision (MAP), NDCG  
**Why**: Only top results matter, ranking quality is crucial

### 7. Stock Price Prediction
**Primary Metrics**: RMSE or MAE  
**Secondary**: R², directional accuracy  
**Why**: Magnitude of error matters for financial decisions

### 8. Real Estate Price Prediction
**Primary Metrics**: MAPE, R²  
**Secondary**: MAE for interpretability  
**Why**: Percentage errors matter more than absolute errors across different price ranges

## Multi-Class Classification Metrics

### Macro-Average
Calculate metric for each class separately, then average  
**Use when**: All classes equally important

### Weighted-Average
Weight each class's metric by its frequency  
**Use when**: Class imbalance exists but all samples equally important

### Micro-Average
Aggregate all samples then calculate metric  
**Use when**: Each sample equally important regardless of class

## Decision Framework

1. **Is it classification or regression?**
   - Classification → Continue to step 2
   - Regression → Use MAE/RMSE/R²/MAPE based on error sensitivity

2. **Are classes balanced?**
   - Yes → Accuracy is meaningful, also use F1
   - No → Avoid accuracy, use F1, AUC-PR, or class-specific metrics

3. **What's the cost of errors?**
   - False Positives costly → Optimize Precision
   - False Negatives costly → Optimize Recall
   - Both costly → Optimize F1 or use cost-sensitive learning

4. **Do you need a single threshold?**
   - Yes → Use threshold-dependent metrics (Precision, Recall, F1)
   - No → Use threshold-independent metrics (AUC-ROC, AUC-PR)

## Interview Tips

- Always start by asking about the business context and costs of different errors
- Mention class imbalance considerations
- Suggest using multiple metrics, not just one
- Be prepared to explain why accuracy alone is often insufficient
- Know how to move from metrics to actionable thresholds

## Common Pitfalls to Avoid

1. Using accuracy on imbalanced datasets
2. Ignoring the business context when choosing metrics
3. Optimizing for a single metric without considering trade-offs
4. Not considering the operating threshold when using AUC
5. Using MAPE when data contains zeros or near-zeros
6. Comparing R² across different datasets
7. Not distinguishing between training and validation metrics
