# Predictive Maintenance: Estimating the Remaining Useful Life of Hard Drives

## Problem

Predict the Remaining Useful Life (RUL) of a hard drive in days using SMART measurements collected during the first month of the observation period.

## Dataset

Source: [Backblaze Hard Drive Test Data](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data), quarterly datasets containing daily SMART measurements for individual hard drives in Backblaze data centers.

The project uses one quarter (Q1 2025), covering approximately 300,000 unique drives and 309,000 rows per daily file.

## Target Variable and Censoring

RUL is calculated from the end of the first month of the quarter (day 30) until drive failure, if the failure occurs during the remaining approximately 60 days of observation.

The main challenge is that most drives **do not fail** within the observation period. For these drives, we only know that they survived for at least another 60 days, rather than their exact RUL.

Instead of assigning an artificial exact target to these drives, a **lower bound of RUL = 60 days** is used. Training relies on a custom loss function that does not penalize overestimation for censored observations but penalizes underestimation.

### Handling Drives That Failed During the First Month

Approximately 377 of the 1,067 drives that failed during the quarter failed within the first month, before a complete 30-day feature window could be collected.

These drives were excluded from the training dataset because their feature histories were not comparable in completeness to those of other drives. This reduces the available number of failure examples to approximately 690.

## Feature Engineering

Features are constructed from raw SMART measurements (`smart_*_raw`) collected during the first month of the quarter.

Measurements are aggregated at three temporal scales—daily, weekly, and monthly—to capture both short-term signals and longer-term degradation trends.

The following features were excluded or transformed:

- **Normalized SMART attributes (`smart_*_normalized`):** Only raw measurements were retained to avoid duplicating information.
- **Infrastructure identifiers (`datacenter`, `cluster_id`, `vault_id`, `pod_id`, `pod_slot_num`):** These were excluded because they are specific to Backblaze's infrastructure and may not generalize to other drive populations. The hypothesis that failures within the same data center or cluster may be correlated due to shared power, cooling, or vibration was not empirically tested in this version of the project.
- **Rare drive models:** Models represented by fewer than approximately 500 drives in the training set were grouped into an `other` category to avoid introducing a large number of sparsely populated one-hot encoded features.

## Train / Validation / Test Split

The dataset was split into training, validation, and test sets in a **70/15/15 ratio**, stratified by the failure indicator (`failure`), without accounting for time.

A known limitation is that the split does not account for potential grouping of drives by data center or cluster. Group-based splitting could provide a more conservative and realistic performance estimate if failures are correlated within the same physical infrastructure.

## Loss Function

Standard MSE cannot be applied directly because of censoring. A custom **Censored Mean Squared Error (CMSE)** was implemented:

- **Drives with known exact RUL:** Standard squared error, penalizing both overestimation and underestimation.
- **Censored drives (RUL lower bound = 60 days):** An error is incurred only when the prediction falls below the lower bound. Overestimation is not penalized.

The loss function was implemented as:

- An evaluation metric for reporting results and comparing models.
- A custom training objective with explicitly calculated gradients and Hessians for CatBoost (`calc_ders_range`) and XGBoost. Separate implementations were required for XGBoost's low-level and scikit-learn-compatible interfaces because their function signatures differ.

Additionally, observation weighting (`sample_weight`) was used to address the rarity of failure examples. This follows the same principle as `scale_pos_weight` in the fraud detection project, but uses the general-purpose `sample_weight` mechanism because `scale_pos_weight` is specific to binary classification and has no direct equivalent in regression.

### Experiment with Relative Weighting (`sample_weight / target`)

An additional experiment multiplied sample weights by `1/RUL` to prioritize accuracy for drives approaching failure, where an error of several days is more critical than the same error for a drive with a longer remaining lifetime.

The results were confirmed on a held-out validation set:

| Metric | Before | After |
|---|---:|---:|
| Mean absolute error, close-to-failure drives (RUL ≤ 15 days) | 28.9 | 18.8 |
| Mean absolute error, remaining RUL > 15 days | 12.7 | 14.1 |

The improvement in accuracy for drives close to failure came at the cost of a slight deterioration for drives with longer remaining lifetimes.

The final decision was to include this weighting strategy in the final model.

## Models and Results

Three model families were compared using the same training and validation sets:

| Model | Validation CMSE (without relative weighting) |
|---|---:|
| Linear Regression / Ridge / ElasticNet | ~130 (after addressing overfitting through regularization and feature standardization) |
| CatBoost | ~56 |
| XGBoost | ~43 |

**An important methodological observation:** Comparing models solely by CMSE can be misleading.

Under certain conditions, linear regression achieved a deceptively low CMSE of approximately 2.2 by predicting values close to 60 days for nearly all observations.

Due to the structure of the censored loss, this strategy incurs almost no penalty for the vast majority of censored observations.

The model's actual discriminative ability—whether it distinguishes failed drives from healthy ones—was therefore evaluated separately by comparing prediction distributions across these groups rather than relying exclusively on a single aggregate metric.

**Final model: XGBoost**, with the following hyperparameters:

- `learning_rate=0.3`
- `max_depth=6`
- `min_child_weight=1`
- `reg_lambda=10`

The model was trained using the custom censored objective and:

`sample_weight = class_balance_weight / RUL`

## Test Evaluation

The test set was evaluated only once, after the model and hyperparameters had been finalized.

| Metric | Validation | Test |
|---|---:|---:|
| CMSE (with relative weighting) | ~484 | 541.6 |
| MAE, close-to-failure drives (RUL ≤ 15 days) | 18.8 | 13.6 |
| MAE, remaining RUL > 15 days | 14.1 | 12.3 |

**CMSE was found to be misleading as the primary metric and was not used for the final assessment.**

The histogram of predictions for censored drives reveals a consistent pattern: predictions peak around 40 days, and nearly all fall below the known lower bound of 60 days.

Because censored MSE specifically penalizes underestimation for these observations, this systematic bias substantially increases CMSE without necessarily implying a proportionally large absolute error.

**MAE was considered a more informative and interpretable metric for this task.** It is approximately 13 days (about 20% of the 60-day observation window) and shows little difference between failed and healthy drives.

## Repository Structure

```text
notebooks/01_eda.ipynb          # Data loading, feature aggregation, target construction, baseline
notebooks/02_modeling.ipynb     # Custom loss, CatBoost/XGBoost training and hyperparameter tuning
notebooks/03_evaluation.ipynb   # Single test evaluation and final metrics
src/losses.py                   # Censored MSE evaluation metrics and custom objectives for both libraries
data/                           # Train/validation/test datasets
models/                         # Saved final XGBoost model (joblib)
```

## Known Limitations and Future Improvements

1. **Censored MSE vs. survival analysis.** The current approach is a practical compromise but does not replace specialized time-to-event models, such as Cox proportional hazards or accelerated failure time models, which are explicitly designed to handle censoring.

2. **Limited accuracy for drives approaching failure.** The absolute error remains approximately 18–19 days on the validation set even after relative weighting. This may be caused by the small number of examples in this range and the high variability of individual drive degradation patterns.

3. **No infrastructure-aware splitting.** The current train/validation/test split does not account for grouping by data center or cluster. This could lead to overly optimistic performance estimates if failures within the same physical infrastructure are correlated. This hypothesis was not empirically tested in the current version.

4. **Only one quarter of data.** Extending the observation period would increase the number of available failure examples but would require redesigning the data ingestion pipeline. The current pipeline cannot scale to the full Backblaze dataset without streaming data processing.

5. **Exclusion of drives that failed during the first month.** Approximately 377 of the 1,067 failed drives were excluded, resulting in the loss of potentially valuable information from the rarest and most critical subset of the data.

6. **Incomplete Numba optimization of the CatBoost custom objective.** The vectorized implementation was incompatible with CatBoost's automatic JIT compilation. As a result, `calc_ders_range` had to be rewritten using element-wise operations, which are less efficient in pure Python but compatible with JIT compilation.