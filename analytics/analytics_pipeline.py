import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# SETUP
# ============================================================

warnings.filterwarnings("ignore")

BASE_DIR = (
    os.path.dirname(os.path.abspath(__file__))
    if "__file__" in locals()
    else os.getcwd()
)

CHARTS_DIR = os.path.join(BASE_DIR, "charts")

TITANIC_CSV = os.path.join(BASE_DIR, "titanic.csv")

BEST_PIPELINE_PATH = os.path.join(
    BASE_DIR, "titanic_best_pipeline.pkl"
)

TUNED_RF_PIPELINE_PATH = os.path.join(
    BASE_DIR, "titanic_tuned_rf_pipeline.pkl"
)

README_PATH = os.path.join(BASE_DIR, "README.md")

os.makedirs(CHARTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def save_current_figure(filename):
    path = os.path.join(CHARTS_DIR, filename)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    return path


def make_preprocessor(numeric_columns, categorical_columns):

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ]
    )


def classification_metrics(model, X_test, y_test):

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "auc": roc_auc_score(
            y_test,
            probabilities,
        ),
        "predictions": predictions,
        "probabilities": probabilities,
    }


def markdown_table(dataframe):

    df = dataframe.copy()

    lines = []

    header = "| " + " | ".join(
        str(c) for c in df.columns
    ) + " |"

    separator = "| " + " | ".join(
        ["---"] * len(df.columns)
    ) + " |"

    lines.append(header)
    lines.append(separator)

    for _, row in df.iterrows():

        values = []

        for value in row:

            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.4f}")

            else:
                values.append(str(value))

        lines.append(
            "| " + " | ".join(values) + " |"
        )

    return "\n".join(lines)


# ============================================================
# TASK 1
# DATASET LOADING AND INITIAL PROFILING
# ============================================================

print("\n" + "=" * 70)
print("TASK 1 - DATASET LOADING AND INITIAL PROFILING")
print("=" * 70)

# IMPORTANT:
# This is the ONLY sns.load_dataset("titanic") call
# in the complete module.

df = sns.load_dataset("titanic")

# Required immediate offline fallback
df.to_csv(TITANIC_CSV, index=False)

print("\nDataset loaded successfully.")
print(f"Offline fallback saved to: {TITANIC_CSV}")

print("\n--- DATASET SHAPE ---")
print(df.shape)

print("\n--- DATASET INFO ---")
df.info()

print("\n--- DESCRIPTIVE STATISTICS ---")
print(df.describe())

print("\n--- FIRST 5 ROWS ---")
print(df.head())


# ============================================================
# TASK 1 - MISSING VALUE PERCENTAGES
# ============================================================

missing_counts = df.isnull().sum()

missing_percentages = (
    missing_counts / len(df)
) * 100

missing_report = pd.DataFrame(
    {
        "Missing Count": missing_counts,
        "Missing Percentage": missing_percentages,
    }
)

missing_report = missing_report[
    missing_report["Missing Count"] > 0
].sort_values(
    "Missing Percentage",
    ascending=False,
)

print("\n--- MISSING VALUE REPORT ---")
print(
    missing_report.to_string(
        float_format=lambda x: f"{x:.2f}"
    )
)


# ============================================================
# TASK 2
# MISSING VALUE HANDLING
# ============================================================

print("\n" + "=" * 70)
print("TASK 2 - MISSING VALUE HANDLING")
print("=" * 70)

df_clean = df.copy()

cleaning_log = []

for column in missing_report.index:

    percentage = missing_percentages[column]

    if percentage < 5:

        df_clean = df_clean.dropna(
            subset=[column]
        )

        message = (
            f"{column}: {percentage:.2f}% missing. "
            f"Below 5%, therefore affected rows were dropped."
        )

        cleaning_log.append(message)

        print(message)

    elif 5 <= percentage <= 30:

        if pd.api.types.is_numeric_dtype(
            df_clean[column]
        ):

            median_value = df_clean[column].median()

            df_clean[column] = (
                df_clean[column].fillna(
                    median_value
                )
            )

            message = (
                f"{column}: {percentage:.2f}% missing. "
                f"Between 5% and 30%, therefore "
                f"median imputation was applied."
            )

        else:

            mode_value = df_clean[column].mode()[0]

            df_clean[column] = (
                df_clean[column].fillna(
                    mode_value
                )
            )

            message = (
                f"{column}: {percentage:.2f}% missing. "
                f"Between 5% and 30%, therefore "
                f"mode imputation was applied."
            )

        cleaning_log.append(message)

        print(message)

    else:

        # More than 30% missing:
        # imputation may be unreliable.
        # We explicitly drop the column.

        df_clean = df_clean.drop(
            columns=[column]
        )

        message = (
            f"{column}: {percentage:.2f}% missing. "
            f"Above 30%, therefore the column was "
            f"dropped because imputation could introduce "
            f"substantial noise."
        )

        cleaning_log.append(message)

        print(message)


print("\n--- CLEANED DATASET SHAPE ---")
print(df_clean.shape)

print("\n--- REMAINING MISSING VALUES ---")
print(
    df_clean.isnull().sum()
)

# IMPORTANT:
# titanic.csv remains the required raw offline fallback.
# All subsequent EDA/modeling uses df_clean.


# ============================================================
# TASK 3
# UNIVARIATE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("TASK 3 - UNIVARIATE ANALYSIS")
print("=" * 70)


# -----------------------------
# AGE HISTOGRAM
# -----------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    df_clean["age"],
    bins=30,
    kde=True,
)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")

save_current_figure(
    "age_histogram.png"
)


# -----------------------------
# AGE BOXPLOT
# -----------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    x=df_clean["age"]
)

plt.title("Age Box Plot")
plt.xlabel("Age")

save_current_figure(
    "age_boxplot.png"
)


# -----------------------------
# FARE HISTOGRAM
# -----------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    df_clean["fare"],
    bins=30,
    kde=True,
)

plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Frequency")

save_current_figure(
    "fare_histogram.png"
)


# -----------------------------
# FARE BOXPLOT
# -----------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    x=df_clean["fare"]
)

plt.title("Fare Box Plot")
plt.xlabel("Fare")

save_current_figure(
    "fare_boxplot.png"
)


# -----------------------------
# IQR OUTLIERS
# -----------------------------

def calculate_iqr_outliers(series):

    q1 = series.quantile(0.25)

    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr

    upper_bound = q3 + 1.5 * iqr

    outliers = series[
        (series < lower_bound)
        | (series > upper_bound)
    ]

    return (
        q1,
        q3,
        iqr,
        lower_bound,
        upper_bound,
        len(outliers),
    )


(
    age_q1,
    age_q3,
    age_iqr,
    age_lower,
    age_upper,
    age_outliers,
) = calculate_iqr_outliers(
    df_clean["age"]
)


(
    fare_q1,
    fare_q3,
    fare_iqr,
    fare_lower,
    fare_upper,
    fare_outliers,
) = calculate_iqr_outliers(
    df_clean["fare"]
)


print("\n--- IQR OUTLIER ANALYSIS ---")

print(
    f"Age Q1 = {age_q1:.2f}, "
    f"Q3 = {age_q3:.2f}, "
    f"IQR = {age_iqr:.2f}"
)

print(
    f"Age lower bound = {age_lower:.2f}, "
    f"upper bound = {age_upper:.2f}"
)

print(
    f"Age outlier count = {age_outliers}"
)

print(
    f"\nFare Q1 = {fare_q1:.2f}, "
    f"Q3 = {fare_q3:.2f}, "
    f"IQR = {fare_iqr:.2f}"
)

print(
    f"Fare lower bound = {fare_lower:.2f}, "
    f"upper bound = {fare_upper:.2f}"
)

print(
    f"Fare outlier count = {fare_outliers}"
)


# -----------------------------
# FARE STATISTICS
# -----------------------------

fare_mean = df_clean["fare"].mean()

fare_median = df_clean["fare"].median()

fare_mode = df_clean["fare"].mode().iloc[0]

print("\n--- FARE STATISTICS ---")

print(
    f"Fare Mean   : {fare_mean:.4f}"
)

print(
    f"Fare Median : {fare_median:.4f}"
)

print(
    f"Fare Mode   : {fare_mode:.4f}"
)


if (
    fare_mean
    > fare_median
    > fare_mode
):

    fare_skew_statement = (
        "Fare is right-skewed because "
        "mean > median > mode."
    )

elif (
    fare_mean
    < fare_median
    < fare_mode
):

    fare_skew_statement = (
        "Fare is left-skewed because "
        "mean < median < mode."
    )

else:

    fare_skew_statement = (
        "Fare does not follow a simple "
        "mean-median-mode ordering; "
        "the distribution should be interpreted "
        "using the histogram as well."
    )

print("\n--- FARE SKEWNESS CONCLUSION ---")
print(fare_skew_statement)


# ============================================================
# TASK 4
# BIVARIATE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("TASK 4 - BIVARIATE ANALYSIS")
print("=" * 70)


# -----------------------------
# SURVIVAL BY SEX
# BOOLEAN MASKING
# -----------------------------

female_mask = df_clean["sex"] == "female"

male_mask = df_clean["sex"] == "male"

female_survival_rate = (
    df_clean.loc[
        female_mask,
        "survived"
    ].mean()
)

male_survival_rate = (
    df_clean.loc[
        male_mask,
        "survived"
    ].mean()
)

print("\n--- SURVIVAL RATE BY SEX ---")

print(
    f"Female: {female_survival_rate:.4f} "
    f"({female_survival_rate * 100:.2f}%)"
)

print(
    f"Male: {male_survival_rate:.4f} "
    f"({male_survival_rate * 100:.2f}%)"
)


# -----------------------------
# SURVIVAL BY PCLASS
# BOOLEAN MASKING
# -----------------------------

first_class_mask = df_clean["pclass"] == 1

second_class_mask = df_clean["pclass"] == 2

third_class_mask = df_clean["pclass"] == 3

pclass_survival = {}

for class_number, mask in [
    (1, first_class_mask),
    (2, second_class_mask),
    (3, third_class_mask),
]:

    rate = (
        df_clean.loc[
            mask,
            "survived"
        ].mean()
    )

    pclass_survival[class_number] = rate

print("\n--- SURVIVAL RATE BY PCLASS ---")

for class_number, rate in pclass_survival.items():

    print(
        f"Pclass {class_number}: "
        f"{rate:.4f} "
        f"({rate * 100:.2f}%)"
    )


# -----------------------------
# SEX + PCLASS
# BOOLEAN MASKING WITH &
# -----------------------------

print("\n--- SURVIVAL RATE BY SEX AND PCLASS ---")

sex_pclass_results = []

for sex_value in ["female", "male"]:

    for class_number in [1, 2, 3]:

        mask = (
            (df_clean["sex"] == sex_value)
            & (df_clean["pclass"] == class_number)
        )

        rate = (
            df_clean.loc[
                mask,
                "survived"
            ].mean()
        )

        sex_pclass_results.append(
            {
                "Sex": sex_value,
                "Pclass": class_number,
                "Survival Rate": rate,
            }
        )

        print(
            f"{sex_value}, Pclass {class_number}: "
            f"{rate:.4f} "
            f"({rate * 100:.2f}%)"
        )


sex_pclass_df = pd.DataFrame(
    sex_pclass_results
)


# ============================================================
# TASK 4
# EXACT 6 x 6 CORRELATION MATRIX
# ============================================================

correlation_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare",
]

correlation_matrix = (
    df_clean[
        correlation_columns
    ].corr()
)

print("\n--- EXACT 6 x 6 CORRELATION MATRIX ---")

print(
    correlation_matrix.round(4)
)


plt.figure(figsize=(9, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="vlag",
    vmin=-1,
    vmax=1,
    square=True,
)

plt.title(
    "Correlation Heatmap - Required Six Numeric Columns"
)

save_current_figure(
    "correlation_heatmap.png"
)


# -----------------------------
# TWO STRONGEST CORRELATIONS
# -----------------------------

pairs = []

for i in range(
    len(correlation_columns)
):

    for j in range(i + 1, len(correlation_columns)):

        col1 = correlation_columns[i]

        col2 = correlation_columns[j]

        corr_value = correlation_matrix.loc[
            col1,
            col2,
        ]

        pairs.append(
            (
                col1,
                col2,
                corr_value,
                abs(corr_value),
            )
        )

pairs = sorted(
    pairs,
    key=lambda x: x[3],
    reverse=True,
)

strongest_pair_1 = pairs[0]

strongest_pair_2 = pairs[1]

print("\n--- TWO STRONGEST CORRELATIONS ---")

print(
    f"1. {strongest_pair_1[0]} vs "
    f"{strongest_pair_1[1]}: "
    f"{strongest_pair_1[2]:.4f}"
)

print(
    f"2. {strongest_pair_2[0]} vs "
    f"{strongest_pair_2[1]}: "
    f"{strongest_pair_2[2]:.4f}"
)


# ============================================================
# TASK 5
# MULTIVARIATE DATA STORY
# ============================================================


# -----------------------------
# CHART 1 - SURVIVAL BY SEX
# -----------------------------

plt.figure(figsize=(7, 5))

sns.barplot(
    data=df_clean,
    x="sex",
    y="survived",
    errorbar=None,
)

plt.title(
    "Survival Rate by Sex"
)

plt.ylabel("Survival Rate")

save_current_figure(
    "survival_by_sex.png"
)


# -----------------------------
# CHART 2 - SURVIVAL BY CLASS
# -----------------------------

plt.figure(figsize=(7, 5))

sns.barplot(
    data=df_clean,
    x="pclass",
    y="survived",
    errorbar=None,
)

plt.title(
    "Survival Rate by Passenger Class"
)

plt.ylabel("Survival Rate")

save_current_figure(
    "survival_by_pclass.png"
)


# -----------------------------
# CHART 3 - SEX + CLASS
# -----------------------------

plt.figure(figsize=(8, 5))

sns.barplot(
    data=df_clean,
    x="pclass",
    y="survived",
    hue="sex",
    errorbar=None,
)

plt.title(
    "Survival Rate by Sex and Passenger Class"
)

plt.ylabel("Survival Rate")

save_current_figure(
    "survival_by_sex_pclass.png"
)


# -----------------------------
# CHART 4 - AGE VS SURVIVAL
# -----------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df_clean,
    x="survived",
    y="age",
)

plt.title(
    "Age Distribution by Survival Status"
)

save_current_figure(
    "age_survival_boxplot.png"
)


# -----------------------------
# CHART 5 - FARE VS SURVIVAL
# -----------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df_clean,
    x="survived",
    y="fare",
)

plt.title(
    "Fare Distribution by Survival Status"
)

save_current_figure(
    "fare_survival_boxplot.png"
)


# -----------------------------
# CHART 6 - SURVIVAL STORY
# -----------------------------

story_df = (
    df_clean.groupby(
        ["pclass", "sex"]
    )["survived"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(9, 5))

sns.barplot(
    data=story_df,
    x="pclass",
    y="survived",
    hue="sex",
    errorbar=None,
)

plt.title(
    "Multivariate Survival Story: Sex and Class"
)

plt.ylabel("Survival Rate")

save_current_figure(
    "survival_story.png"
)


# ============================================================
# TASK 6
# EDA STANDARDIZATION SANITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("TASK 6 - AGE AND FARE STANDARDIZATION CHECK")
print("=" * 70)


eda_standardized = df_clean[
    ["age", "fare"]
].copy()

for column in ["age", "fare"]:

    mean_value = eda_standardized[
        column
    ].mean()

    std_value = eda_standardized[
        column
    ].std()

    eda_standardized[
        f"{column}_z"
    ] = (
        eda_standardized[column]
        - mean_value
    ) / std_value


print("\n--- BEFORE STANDARDIZATION ---")

print(
    df_clean[
        ["age", "fare"]
    ].agg(
        ["mean", "std"]
    )
)


print("\n--- AFTER STANDARDIZATION ---")

print(
    eda_standardized[
        ["age_z", "fare_z"]
    ].agg(
        ["mean", "std"]
    )
)


# Age standardized plot

plt.figure(figsize=(8, 5))

sns.histplot(
    eda_standardized["age_z"],
    bins=30,
    kde=True,
)

plt.title(
    "Standardized Age Distribution"
)

plt.xlabel("Age Z-score")

save_current_figure(
    "age_standardization.png"
)


# Fare standardized plot

plt.figure(figsize=(8, 5))

sns.histplot(
    eda_standardized["fare_z"],
    bins=30,
    kde=True,
)

plt.title(
    "Standardized Fare Distribution"
)

plt.xlabel("Fare Z-score")

save_current_figure(
    "fare_standardization.png"
)


# ============================================================
# TASK 7
# TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("TASK 7 - STRATIFIED TRAIN TEST SPLIT")
print("=" * 70)


X = df_clean[
    [
        "pclass",
        "sex",
        "age",
        "sibsp",
        "parch",
        "fare",
        "embarked",
    ]
].copy()

y = df_clean[
    "survived"
].copy()


print("\n--- CLASS BALANCE BEFORE SPLIT ---")

class_balance = (
    y.value_counts()
    .sort_index()
)

class_balance_percentage = (
    y.value_counts(
        normalize=True
    )
    .sort_index()
    * 100
)

for class_value in class_balance.index:

    print(
        f"Class {class_value}: "
        f"{class_balance[class_value]} rows "
        f"({class_balance_percentage[class_value]:.2f}%)"
    )


print(
    "\nJustification for stratification:"
)

print(
    "The survived target contains two classes. "
    "A stratified split preserves approximately "
    "the same class proportions in both training "
    "and test sets, reducing the risk that one split "
    "has a distorted survival distribution."
)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


print(
    f"\nTraining rows: {len(X_train)}"
)

print(
    f"Testing rows: {len(X_test)}"
)


# ============================================================
# TASK 8
# PREPROCESSING
# ============================================================

numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare",
]

categorical_features = [
    "sex",
    "embarked",
]


# ============================================================
# TASK 9
# THREE CLASSIFIERS
# ============================================================

print("\n" + "=" * 70)
print("TASK 9 - THREE CLASSIFIERS")
print("=" * 70)


models = {

    "Logistic Regression": Pipeline(
        [
            (
                "preprocessor",
                make_preprocessor(
                    numeric_features,
                    categorical_features,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    ),

    "Decision Tree": Pipeline(
        [
            (
                "preprocessor",
                make_preprocessor(
                    numeric_features,
                    categorical_features,
                ),
            ),
            (
                "classifier",
                DecisionTreeClassifier(
                    random_state=42,
                    max_depth=5,
                ),
            ),
        ]
    ),

    "Random Forest": Pipeline(
        [
            (
                "preprocessor",
                make_preprocessor(
                    numeric_features,
                    categorical_features,
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=42,
                ),
            ),
        ]
    ),
}


classification_results = []

model_metrics_store = {}


for model_name, model in models.items():

    print(
        f"\nTraining {model_name}..."
    )

    model.fit(
        X_train,
        y_train,
    )

    metrics = classification_metrics(
        model,
        X_test,
        y_test,
    )

    model_metrics_store[
        model_name
    ] = metrics

    classification_results.append(
        {
            "Model": model_name,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1"],
            "AUC": metrics["auc"],
        }
    )


classification_table = pd.DataFrame(
    classification_results
)

print(
    "\n--- THREE CLASSIFIER COMPARISON ---"
)

print(
    classification_table.to_string(
        index=False
    )
)


# ============================================================
# TASK 9
# DECISION TREE VISUALIZATION
# ============================================================

print("\n--- DECISION TREE VISUALIZATION ---")


decision_tree_model = models[
    "Decision Tree"
]

preprocessor_dt = (
    decision_tree_model
    .named_steps["preprocessor"]
)

tree_model = (
    decision_tree_model
    .named_steps["classifier"]
)

feature_names = (
    preprocessor_dt
    .get_feature_names_out()
)

plt.figure(
    figsize=(22, 12)
)

plot_tree(
    tree_model,
    feature_names=feature_names,
    class_names=[
        "Not Survived",
        "Survived",
    ],
    filled=True,
    rounded=True,
    fontsize=7,
)

plt.title(
    "Decision Tree for Titanic Survival"
)

save_current_figure(
    "decision_tree.png"
)


# ============================================================
# TASK 10
# CONFUSION MATRICES + ROC CURVES
# ============================================================

print("\n" + "=" * 70)
print("TASK 10 - CONFUSION MATRICES AND ROC CURVES")
print("=" * 70)


for model_name, metrics in model_metrics_store.items():

    cm = confusion_matrix(
        y_test,
        metrics["predictions"],
    )

    print(
        f"\n--- {model_name} CONFUSION MATRIX ---"
    )

    print(cm)

    print(
        f"Accuracy : {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: {metrics['precision']:.4f}"
    )

    print(
        f"Recall   : {metrics['recall']:.4f}"
    )

    print(
        f"F1       : {metrics['f1']:.4f}"
    )

    print(
        f"AUC      : {metrics['auc']:.4f}"
    )


# ROC CURVES

plt.figure(
    figsize=(9, 7)
)

for model_name, metrics in model_metrics_store.items():

    fpr, tpr, _ = roc_curve(
        y_test,
        metrics["probabilities"],
    )

    plt.plot(
        fpr,
        tpr,
        label=(
            f"{model_name} "
            f"(AUC={metrics['auc']:.3f})"
        ),
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier",
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curves - Three Classifiers"
)

plt.legend()

save_current_figure(
    "roc_curves.png"
)


# ============================================================
# TASK 11
# IMBALANCE COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("TASK 11 - IMBALANCE HANDLING")
print("=" * 70)


imbalance_models = {

    "Baseline": Pipeline(
        [
            (
                "preprocessor",
                make_preprocessor(
                    numeric_features,
                    categorical_features,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    ),

    "Class Weight Balanced": Pipeline(
        [
            (
                "preprocessor",
                make_preprocessor(
                    numeric_features,
                    categorical_features,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    ),

    "SMOTE": ImbPipeline(
        [
            (
                "preprocessor",
                make_preprocessor(
                    numeric_features,
                    categorical_features,
                ),
            ),
            (
                "smote",
                SMOTE(
                    random_state=42
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    ),
}


imbalance_results = []


for method_name, model in imbalance_models.items():

    model.fit(
        X_train,
        y_train,
    )

    metrics = classification_metrics(
        model,
        X_test,
        y_test,
    )

    imbalance_results.append(
        {
            "Strategy": method_name,
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1"],
        }
    )


imbalance_df = pd.DataFrame(
    imbalance_results
)


print(
    "\n--- IMBALANCE COMPARISON ---"
)

print(
    imbalance_df.to_string(
        index=False
    )
)


best_imbalance_row = (
    imbalance_df.loc[
        imbalance_df["F1"].idxmax()
    ]
)

imbalance_conclusion = (
    f"Among the three imbalance strategies, "
    f"{best_imbalance_row['Strategy']} produced the "
    f"highest F1 score of "
    f"{best_imbalance_row['F1']:.4f}. "
    f"The comparison should be interpreted together "
    f"with precision and recall because F1 balances "
    f"these two measures. SMOTE was applied only "
    f"inside the training pipeline, so synthetic samples "
    f"were not created from the test set."
)

print(
    "\n--- IMBALANCE CONCLUSION ---"
)

print(
    imbalance_conclusion
)


# ============================================================
# TASK 12
# RANDOM FOREST GRID SEARCH + OOB
# ============================================================

print("\n" + "=" * 70)
print("TASK 12 - RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 70)


rf_grid_pipeline = Pipeline(
    [
        (
            "preprocessor",
            make_preprocessor(
                numeric_features,
                categorical_features,
            ),
        ),
        (
            "classifier",
            RandomForestClassifier(
                random_state=42,
                oob_score=True,
                bootstrap=True,
            ),
        ),
    ]
)


param_grid = {

    "classifier__n_estimators": [
        50,
        100,
        200,
    ],

    "classifier__max_depth": [
        3,
        5,
        10,
        None,
    ],

    "classifier__max_features": [
        "sqrt",
        "log2",
        None,
    ],
}


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


grid_search = GridSearchCV(
    estimator=rf_grid_pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="f1",
    n_jobs=-1,
)


print(
    "Running GridSearchCV..."
)

grid_search.fit(
    X_train,
    y_train,
)


best_rf_model = (
    grid_search.best_estimator_
)


print(
    "\nBest Parameters:"
)

print(
    grid_search.best_params_
)


print(
    f"\nBest Cross-Validation F1: "
    f"{grid_search.best_score_:.4f}"
)


# OOB SCORE

tuned_rf_estimator = (
    best_rf_model
    .named_steps["classifier"]
)

oob_score = (
    tuned_rf_estimator.oob_score_
)


print(
    f"OOB Score: {oob_score:.4f}"
)


# Save tuned complete pipeline

joblib.dump(
    best_rf_model,
    TUNED_RF_PIPELINE_PATH,
)

print(
    f"Tuned complete RF pipeline saved to:\n"
    f"{TUNED_RF_PIPELINE_PATH}"
)


# ============================================================
# TASK 13
# REGRESSION SIDE TASK
# ============================================================

print("\n" + "=" * 70)
print("TASK 13 - MULTIVARIATE LINEAR REGRESSION")
print("=" * 70)


regression_features = [
    column
    for column in df_clean.columns
    if column != "fare"
]


# Use numeric predictors for a clean
# multivariate linear regression.

regression_numeric_features = (
    df_clean[
        regression_features
    ]
    .select_dtypes(
        include=np.number
    )
    .columns
    .tolist()
)


# Do not use fare itself.

regression_numeric_features = [
    column
    for column in regression_numeric_features
    if column != "fare"
]


X_reg = df_clean[
    regression_numeric_features
].copy()

y_reg = df_clean[
    "fare"
].copy()


X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42,
)


regression_pipeline = Pipeline(
    [
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "regressor",
            LinearRegression(),
        ),
    ]
)


regression_pipeline.fit(
    X_reg_train,
    y_reg_train,
)


y_reg_pred = (
    regression_pipeline.predict(
        X_reg_test
    )
)


mae = mean_absolute_error(
    y_reg_test,
    y_reg_pred,
)

mse = mean_squared_error(
    y_reg_test,
    y_reg_pred,
)

rmse = np.sqrt(mse)

r2 = (
    1
    - (
        np.sum(
            (y_reg_test - y_reg_pred) ** 2
        )
        /
        np.sum(
            (y_reg_test - y_reg_test.mean()) ** 2
        )
    )
)


n = len(y_reg_test)

p = X_reg_test.shape[1]

if n - p - 1 != 0:

    adjusted_r2 = (
        1
        -
        (
            (1 - r2)
            * (n - 1)
            /
            (n - p - 1)
        )
    )

else:

    adjusted_r2 = np.nan


print(
    "\n--- REGRESSION METRICS ---"
)

print(
    f"MAE       : {mae:.4f}"
)

print(
    f"RMSE      : {rmse:.4f}"
)

print(
    f"R²        : {r2:.4f}"
)

print(
    f"Adjusted R²: {adjusted_r2:.4f}"
)


# -----------------------------
# RESIDUAL PLOT
# -----------------------------

residuals = (
    y_reg_test
    - y_reg_pred
)


plt.figure(
    figsize=(9, 6)
)

plt.scatter(
    y_reg_pred,
    residuals,
    alpha=0.6,
)

plt.axhline(
    y=0,
    linestyle="--",
)

plt.xlabel(
    "Predicted Fare"
)

plt.ylabel(
    "Residual"
)

plt.title(
    "Residual Plot - Fare Regression"
)

save_current_figure(
    "fare_residual_plot.png"
)


# Simple exploratory heteroscedasticity check

abs_residual_correlation = np.corrcoef(
    y_reg_pred,
    np.abs(residuals),
)[0, 1]


if abs_residual_correlation > 0.30:

    heteroscedasticity_conclusion = (
        "The residual plot suggests possible "
        "heteroscedasticity because the magnitude "
        "of residuals increases with predicted fare."
    )

else:

    heteroscedasticity_conclusion = (
        "The residual plot does not show strong "
        "evidence of heteroscedasticity based on "
        "the observed residual spread. Residuals "
        "should ideally remain randomly scattered "
        "around zero with roughly constant spread."
    )


print(
    "\n--- HETEROSCEDASTICITY CONCLUSION ---"
)

print(
    heteroscedasticity_conclusion
)


# ============================================================
# TASK 14
# FINAL MODEL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("TASK 14 - FINAL MODEL COMPARISON")
print("=" * 70)


# Classification metric group

classification_final = classification_table.copy()

classification_final = (
    classification_final[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "AUC",
        ]
    ]
)


print(
    "\n--- CLASSIFICATION METRICS ---"
)

print(
    classification_final.to_string(
        index=False
    )
)


# Regression metric group

regression_final = pd.DataFrame(
    [
        {
            "Regression Model": "Multivariate Linear Regression",
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Adjusted R2": adjusted_r2,
        }
    ]
)


print(
    "\n--- REGRESSION METRICS ---"
)

print(
    regression_final.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# Choose deployment candidate based on F1
# ------------------------------------------------------------

best_classifier_row = (
    classification_table.loc[
        classification_table["F1"].idxmax()
    ]
)

deployment_model_name = (
    best_classifier_row["Model"]
)

deployment_f1 = (
    best_classifier_row["F1"]
)

deployment_auc = (
    best_classifier_row["AUC"]
)

deployment_precision = (
    best_classifier_row["Precision"]
)

deployment_recall = (
    best_classifier_row["Recall"]
)

deployment_accuracy = (
    best_classifier_row["Accuracy"]
)


final_recommendation = (
    f"Based on the held-out test-set results, "
    f"{deployment_model_name} is the classifier selected "
    f"for deployment in this project because it produced "
    f"the highest F1 score among the three evaluated "
    f"classifiers. Its accuracy was {deployment_accuracy:.4f}, "
    f"precision was {deployment_precision:.4f}, "
    f"recall was {deployment_recall:.4f}, "
    f"F1 was {deployment_f1:.4f}, and AUC was "
    f"{deployment_auc:.4f}. These metrics provide a "
    f"balanced view of classification performance rather "
    f"than relying on accuracy alone. The final choice "
    f"should therefore be understood as a metric-based "
    f"selection using the specified held-out test results."
)


print(
    "\n--- FINAL CLASSIFIER RECOMMENDATION ---"
)

print(
    final_recommendation
)


# ============================================================
# TASK 15
# SAVE COMPLETE PIPELINE
# ============================================================

print("\n" + "=" * 70)
print("TASK 15 - SAVE AND RELOAD COMPLETE PIPELINE")
print("=" * 70)


# Use the selected classifier as the final complete pipeline.

if deployment_model_name == "Logistic Regression":

    full_pipeline = models[
        "Logistic Regression"
    ]

elif deployment_model_name == "Decision Tree":

    full_pipeline = models[
        "Decision Tree"
    ]

else:

    full_pipeline = models[
        "Random Forest"
    ]


# Refit selected complete pipeline on training data only.

full_pipeline.fit(
    X_train,
    y_train,
)


joblib.dump(
    full_pipeline,
    BEST_PIPELINE_PATH,
)


print(
    f"Complete preprocessing + estimator pipeline "
    f"saved to:\n{BEST_PIPELINE_PATH}"
)


# ------------------------------------------------------------
# RELOAD SAVED PIPELINE
# ------------------------------------------------------------

loaded_pipeline = joblib.load(
    BEST_PIPELINE_PATH
)


raw_new_input = X_test.iloc[
    [0]
].copy()


reloaded_prediction = (
    loaded_pipeline.predict(
        raw_new_input
    )
)


print(
    "\n--- RELOADED PIPELINE TEST ---"
)

print(
    "Raw input:"
)

print(
    raw_new_input
)

print(
    f"\nPrediction from reloaded pipeline: "
    f"{reloaded_prediction[0]}"
)

print(
    f"Actual test label: "
    f"{y_test.iloc[0]}"
)


# ============================================================
# README GENERATION
# ============================================================

print("\n" + "=" * 70)
print("CREATING README.md")
print("=" * 70)


readme_text = f"""
# Module 2 — Analytics Pipeline

## Dataset Loading

The Titanic dataset was loaded once using:

`sns.load_dataset("titanic")`

The loaded raw DataFrame was immediately saved as:

`titanic.csv`

This file serves as the offline fallback.

## Task 1 — Profiling

Dataset shape:

`{df.shape}`

Missing values were measured as percentages for every affected column.

### Missing Value Report

{missing_report.round(2).to_string()}

## Task 2 — Cleaning Strategy

The percentage-based threshold rule was applied:

- Below 5% missing: affected rows dropped.
- 5% to 30% missing: imputation applied.
- Above 30% missing: column dropped when imputation was considered unreliable.

### Cleaning Decisions

"""

for item in cleaning_log:

    readme_text += (
        f"- {item}\n"
    )


readme_text += f"""

## Task 3 — Univariate Analysis

### Age Outliers

Using the IQR rule:

- Q1 = {age_q1:.4f}
- Q3 = {age_q3:.4f}
- IQR = {age_iqr:.4f}
- Lower bound = {age_lower:.4f}
- Upper bound = {age_upper:.4f}
- Outlier count = {age_outliers}

### Fare Outliers

Using the IQR rule:

- Q1 = {fare_q1:.4f}
- Q3 = {fare_q3:.4f}
- IQR = {fare_iqr:.4f}
- Lower bound = {fare_lower:.4f}
- Upper bound = {fare_upper:.4f}
- Outlier count = {fare_outliers}

### Fare Statistics

- Mean = {fare_mean:.4f}
- Median = {fare_median:.4f}
- Mode = {fare_mode:.4f}

### Fare Distribution Interpretation

{fare_skew_statement}

The ordering of mean, median and mode was used as the primary interpretation of skewness.

## Task 4 — Bivariate Analysis

### Survival by Sex

Female survival rate = {female_survival_rate:.4f}

Male survival rate = {male_survival_rate:.4f}

The survival rates demonstrate a substantial difference between the two sex groups in this dataset.

### Survival by Passenger Class

"""

for class_number, rate in pclass_survival.items():

    readme_text += (
        f"- Pclass {class_number}: "
        f"{rate:.4f}\n"
    )


readme_text += f"""

### Survival by Sex and Passenger Class

"""

for _, row in sex_pclass_df.iterrows():

    readme_text += (
        f"- {row['Sex']}, Pclass "
        f"{int(row['Pclass'])}: "
        f"{row['Survival Rate']:.4f}\n"
    )


readme_text += f"""

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

1. {strongest_pair_1[0]} and {strongest_pair_1[1]}:
   correlation = {strongest_pair_1[2]:.4f}

2. {strongest_pair_2[0]} and {strongest_pair_2[1]}:
   correlation = {strongest_pair_2[2]:.4f}

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

{classification_table.round(4).to_string(index=False)}

The three classifiers were trained on the same train/test split.

Confusion matrices and ROC curves were generated for all three models.

The Decision Tree was rendered using `plot_tree` with feature names and class names.

## Task 11 — Imbalance Handling

{imbalance_df.round(4).to_string(index=False)}

### Conclusion

{imbalance_conclusion}

## Task 12 — Hyperparameter Tuning

Best Random Forest parameters:

`{grid_search.best_params_}`

Best cross-validation F1:

`{grid_search.best_score_:.4f}`

OOB score:

`{oob_score:.4f}`

The Random Forest was constructed with `oob_score=True`.

## Task 13 — Regression

The regression task predicts `fare` using the other available numeric features.

Regression metrics:

- MAE = {mae:.4f}
- RMSE = {rmse:.4f}
- R² = {r2:.4f}
- Adjusted R² = {adjusted_r2:.4f}

### Residual Plot Interpretation

{heteroscedasticity_conclusion}

The residual plot was saved as `charts/fare_residual_plot.png`.

## Task 14 — Final Model Comparison

### Classification Metrics

{classification_final.round(4).to_string(index=False)}

### Regression Metrics

{regression_final.round(4).to_string(index=False)}

Classification and regression metrics are presented as separate metric groups because they measure different types of predictive tasks and are not directly comparable on one common scale.

### Final Classifier Recommendation

{final_recommendation}

## Task 15 — Saved Pipeline

The final complete preprocessing + estimator pipeline was saved as:

`titanic_best_pipeline.pkl`

The artifact was reloaded using `joblib.load()` and tested using raw, unpreprocessed feature input.

The pipeline therefore contains the preprocessing steps and final estimator together and can perform end-to-end prediction.
"""


with open(
    README_PATH,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        readme_text
    )


print(
    f"README created at:\n{README_PATH}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MODULE 2 PIPELINE COMPLETED")
print("=" * 70)

print(
    "\nGenerated:"
)

print(
    "1. titanic.csv"
)

print(
    "2. charts/*.png"
)

print(
    "3. titanic_best_pipeline.pkl"
)

print(
    "4. titanic_tuned_rf_pipeline.pkl"
)

print(
    "5. README.md"
)

print(
    "\nModule 2 execution finished successfully."
)