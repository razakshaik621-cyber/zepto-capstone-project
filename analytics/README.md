
# Module 2 — Analytics Pipeline

## Dataset Loading

The Titanic dataset was loaded once using:

`sns.load_dataset("titanic")`

The loaded raw DataFrame was immediately saved as:

`titanic.csv`

This file serves as the offline fallback.

## Task 1 — Profiling

Dataset shape:

`(891, 15)`

Missing values were measured as percentages for every affected column.

### Missing Value Report

             Missing Count  Missing Percentage
deck                   688               77.22
age                    177               19.87
embarked                 2                0.22
embark_town              2                0.22

## Task 2 — Cleaning Strategy

The percentage-based threshold rule was applied:

- Below 5% missing: affected rows dropped.
- 5% to 30% missing: imputation applied.
- Above 30% missing: column dropped when imputation was considered unreliable.

### Cleaning Decisions

- deck: 77.22% missing. Above 30%, therefore the column was dropped because imputation could introduce substantial noise.
- age: 19.87% missing. Between 5% and 30%, therefore median imputation was applied.
- embarked: 0.22% missing. Below 5%, therefore affected rows were dropped.
- embark_town: 0.22% missing. Below 5%, therefore affected rows were dropped.


## Task 3 — Univariate Analysis

### Age Outliers

Using the IQR rule:

- Q1 = 22.0000
- Q3 = 35.0000
- IQR = 13.0000
- Lower bound = 2.5000
- Upper bound = 54.5000
- Outlier count = 65

### Fare Outliers

Using the IQR rule:

- Q1 = 7.8958
- Q3 = 31.0000
- IQR = 23.1042
- Lower bound = -26.7605
- Upper bound = 65.6563
- Outlier count = 114

### Fare Statistics

- Mean = 32.0967
- Median = 14.4542
- Mode = 8.0500

### Fare Distribution Interpretation

Fare is right-skewed because mean > median > mode.

The ordering of mean, median and mode was used as the primary interpretation of skewness.

## Task 4 — Bivariate Analysis

### Survival by Sex

Female survival rate = 0.7404

Male survival rate = 0.1889

The survival rates demonstrate a substantial difference between the two sex groups in this dataset.

### Survival by Passenger Class

- Pclass 1: 0.6262
- Pclass 2: 0.4728
- Pclass 3: 0.2424


### Survival by Sex and Passenger Class

- female, Pclass 1: 0.9674
- female, Pclass 2: 0.9211
- female, Pclass 3: 0.5000
- male, Pclass 1: 0.3689
- male, Pclass 2: 0.1574
- male, Pclass 3: 0.1354


### Correlation Matrix

The correlation matrix contains exactly these six columns:

- survived
- pclass
- age
- sibsp
- parch
- fare

The boolean columns `adult_male` and `alone` were excluded because they are derived/redundant flags.

### Two Strongest Correlations

1. pclass and fare:
   correlation = -0.5482

2. sibsp and parch:
   correlation = 0.4145

The first pair has the largest absolute off-diagonal correlation in the required six-column matrix. The second pair has the second-largest absolute off-diagonal correlation.

## Task 5 — Multivariate Data Story

### Chart 1 — Survival by Sex

The chart shows survival probability broken down by sex. The difference indicates that sex was strongly associated with survival in the Titanic data. This relationship is also reflected in the classification features.

### Chart 2 — Survival by Passenger Class

The chart shows survival probability across passenger classes. Survival rates differ across Pclass, indicating that passenger class provides useful information about survival outcomes.

### Chart 3 — Survival by Sex and Passenger Class

Combining sex and Pclass reveals that survival patterns are not explained by one variable alone. The interaction between these two characteristics gives a more detailed view of which passenger groups experienced different survival rates.

### Chart 4 — Age and Survival

The age box plot compares the age distributions of survivors and non-survivors. It helps show whether the age distribution differs between the two outcome groups.

### Chart 5 — Fare and Survival

The fare box plot compares fare distributions across survival status. Differences in fare distributions provide another indication that socioeconomic position was associated with survival.

### Chart 6 — Multivariate Survival Story

The combined sex-and-class chart summarizes the main multivariate pattern. It shows that survival differed across passenger groups defined jointly by sex and passenger class.

## Task 6 — Standardization Sanity Check

The EDA-stage z-score transformation used:

`z = (x - mean) / std`

The transformed age and fare columns have approximately mean 0 and standard deviation 1.

This standardization check was exploratory only and was not used as the modeling preprocessing. The actual modeling pipeline performs training-only preprocessing.

## Task 7 — Stratified Split

The classification target is `survived`.

Stratification was used so that the training and testing sets retain approximately the same class distribution as the original target. This is important because the survival classes are not perfectly balanced.

## Task 8 — Preprocessing

Numeric features use median imputation followed by StandardScaler.

Categorical features use most-frequent imputation followed by one-hot encoding.

The preprocessing is implemented using ColumnTransformer and Pipeline, so preprocessing is fitted on the training data and then applied to the test data through transform operations.

## Task 9 and 10 — Classification Results

              Model  Accuracy  Precision  Recall     F1    AUC
Logistic Regression    0.8146     0.7966  0.6912 0.7402 0.8610
      Decision Tree    0.7640     0.7600  0.5588 0.6441 0.8374
      Random Forest    0.8034     0.7619  0.7059 0.7328 0.8266

The three classifiers were trained on the same train/test split.

Confusion matrices and ROC curves were generated for all three models.

The Decision Tree was rendered using `plot_tree` with feature names and class names.

## Task 11 — Imbalance Handling

             Strategy  Precision  Recall     F1
             Baseline     0.7966  0.6912 0.7402
Class Weight Balanced     0.7083  0.7500 0.7286
                SMOTE     0.7463  0.7353 0.7407

### Conclusion

Among the three imbalance strategies, SMOTE produced the highest F1 score of 0.7407. The comparison should be interpreted together with precision and recall because F1 balances these two measures. SMOTE was applied only inside the training pipeline, so synthetic samples were not created from the test set.

## Task 12 — Hyperparameter Tuning

Best Random Forest parameters:

`{'classifier__max_depth': 10, 'classifier__max_features': None, 'classifier__n_estimators': 50}`

Best cross-validation F1:

`0.7828`

OOB score:

`0.8228`

The Random Forest was constructed with `oob_score=True`.

## Task 13 — Regression

The regression task predicts `fare` using the other available numeric features.

Regression metrics:

- MAE = 20.6700
- RMSE = 42.4424
- R² = 0.3248
- Adjusted R² = 0.3052

### Residual Plot Interpretation

The residual plot suggests possible heteroscedasticity because the magnitude of residuals increases with predicted fare.

The residual plot was saved as `charts/fare_residual_plot.png`.

## Task 14 — Final Model Comparison

### Classification Metrics

              Model  Accuracy  Precision  Recall     F1    AUC
Logistic Regression    0.8146     0.7966  0.6912 0.7402 0.8610
      Decision Tree    0.7640     0.7600  0.5588 0.6441 0.8374
      Random Forest    0.8034     0.7619  0.7059 0.7328 0.8266

### Regression Metrics

              Regression Model   MAE    RMSE     R2  Adjusted R2
Multivariate Linear Regression 20.67 42.4424 0.3248       0.3052

Classification and regression metrics are presented as separate metric groups because they measure different types of predictive tasks and are not directly comparable on one common scale.

### Final Classifier Recommendation

Based on the held-out test-set results, Logistic Regression is the classifier selected for deployment in this project because it produced the highest F1 score among the three evaluated classifiers. Its accuracy was 0.8146, precision was 0.7966, recall was 0.6912, F1 was 0.7402, and AUC was 0.8610. These metrics provide a balanced view of classification performance rather than relying on accuracy alone. The final choice should therefore be understood as a metric-based selection using the specified held-out test results.

## Task 15 — Saved Pipeline

The final complete preprocessing + estimator pipeline was saved as:

`titanic_best_pipeline.pkl`

The artifact was reloaded using `joblib.load()` and tested using raw, unpreprocessed feature input.

The pipeline therefore contains the preprocessing steps and final estimator together and can perform end-to-end prediction.
