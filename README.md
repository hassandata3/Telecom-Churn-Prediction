# Telecom Customer Churn Prediction

Predicts which customers are likely to cancel their service, so retention efforts can be targeted at the right people before they leave.

## Business Value
Losing a customer costs more than keeping one. This model flags at-risk customers in advance, ranked by churn probability, so retention offers (discounts, outreach, plan upgrades) go to the people most likely to need them — instead of being sent blindly to everyone.

## Key Results
- **~77% accuracy**, catching **~70-80% of customers who actually churn**, using a threshold tuned to prioritize recall (better to over-flag a few extra customers than miss real churners).
- Best model: **Logistic Regression** — simple, fast, and outperformed more complex models here.

## What Drives Churn
1. **Tenure** — newer customers churn far more than long-tenured ones.
2. **Contract type** — month-to-month customers churn at ~4x the rate of annual/2-year contract customers.
3. **Billing & internet service type** — higher charges and fiber-optic service correlate with higher churn.



**Takeaway for the business:** incentivizing longer contracts and monitoring newer/high-bill customers closely would likely have the biggest impact on reducing churn.

## How It Would Be Used
Run monthly against the full customer base → output a ranked list of churn probabilities → retention team prioritizes outreach to the highest-risk segment, rather than reacting only after a customer cancels.

## Notes
- Initial version used a different dataset with a data quality issue (one feature was almost perfectly correlated with the outcome, inflating accuracy to 95%+). Caught this during review and switched to a cleaner, industry-standard dataset — a good reminder to sanity-check unusually high scores before trusting them.
- Precision (~54%) is lower than accuracy by design: the model favors catching more real churners over avoiding false alarms, which is the right tradeoff when retention outreach is low-cost.

## Technical Approach
- **Preprocessing:** binary fields mapped to 0/1, `Contract` encoded as an ordinal numeric feature (1/12/24 months) rather than one-hot, since contract length has a natural order. `MultipleLines` and `InternetService` (true multi-category fields) one-hot encoded. Missing `TotalCharges` values imputed with the training-set mean.
- **Models compared:** Logistic Regression, Random Forest, XGBoost — all evaluated at the same classification threshold for a fair comparison.
- **Threshold selection:** tuned via a precision/recall/F1 sweep (0.20–0.50) rather than using the default 0.5, since missing a churner is costlier than a false alarm.
- **Validation:** train/test split with stratification to preserve the churn ratio in both sets.

## Bugs Found & Fixed
- **Data leakage:** scaler/imputer were initially fit on the full dataset before splitting into train/test, leaking test-set information into preprocessing. Fixed by fitting only on the training set and applying `.transform()` (not `.fit_transform()`) to test data.
- **Global find-and-replace bug:** an unscoped `.replace({'No': 0, 'Yes': 1})` across the entire DataFrame silently corrupted multi-category columns (`MultipleLines`, `InternetService`) by converting some but not all of their values. Fixed by scoping replacements to specific binary columns only.
- **Dtype bug:** `.replace()` left binary columns as `object` dtype even though values were correct integers, which caused `LogisticRegression.fit()` to throw `"Unknown label type: unknown"` (surfaced by a pandas `FutureWarning`). Fixed with an explicit `.astype(int)`.
- **Scaling mismatch:** Logistic Regression was trained on scaled data but evaluated on unscaled data in one analysis step, producing distorted probability outputs. Fixed by using the same scaled features consistently for training and inference.

## Tech Stack
Python, pandas, scikit-learn, XGBoost, Gradio (demo interface)
