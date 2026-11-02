import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# BOLD TERMINAL HEADING FUNCTION
def print_heading(title):
    print(f"\n\033[1m{title}\033[0m")

# 1. PATHS
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "healthy_diet_calorie_intake.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "calorie_model.pkl"
)

GRAPH_DIR = os.path.join(
    BASE_DIR,
    "graphs"
)

os.makedirs(
    os.path.dirname(MODEL_PATH),
    exist_ok=True
)

os.makedirs(
    GRAPH_DIR,
    exist_ok=True
)


# 2. LOAD DATASET

print_heading(
    "NUTRIPREDICT - DATASET ANALYSIS & MODEL TRAINING"
)

print("\nLoading Dataset...")

try:
    df = pd.read_csv(DATASET_PATH)

except FileNotFoundError:
    print("\nERROR: Dataset file not found!")
    print("Expected location:")
    print(DATASET_PATH)
    exit()

print("\nDataset loaded successfully!")

print(
    f"Dataset Shape: "
    f"{df.shape[0]} rows × {df.shape[1]} columns"
)

print("\nColumns:")

for column in df.columns:
    print(" -", column)


# 3. DATA TYPES

print_heading(
    "3. DATA TYPES"
)

print(df.dtypes)


# 4. MISSING VALUE CHECK

print_heading(
    "4. MISSING VALUE CHECK - EACH PARAMETER"
)

missing_values = df.isnull().sum()

missing_report = pd.DataFrame({
    "Parameter": missing_values.index,
    "Missing_Records": missing_values.values
})

for _, row in missing_report.iterrows():

    print(
        f"{row['Parameter']:<30}"
        f"{int(row['Missing_Records'])}"
    )

total_missing = int(
    missing_values.sum()
)

print(
    "\nTotal Missing Values:",
    total_missing
)


# 5. DUPLICATE RECORD CHECK

print_heading(
    "5. DUPLICATE RECORD CHECK"
)

duplicate_count = int(
    df.duplicated().sum()
)

print(
    "Total Duplicate Records:",
    duplicate_count
)

if duplicate_count == 0:

    print(
        "No duplicate records found."
    )

else:

    print(
        "Duplicate records found."
    )


# 6. INVALID VALUE CHECK

print_heading(
    "6. INVALID VALUE CHECK - EACH NUMERIC PARAMETER"
)

numeric_columns = [
    "Age",
    "Height_cm",
    "Weight_kg",
    "BMI",
    "Daily_Calorie_Requirement",
    "Daily_Calorie_Consumed",
    "Protein_Intake_g",
    "Carbohydrate_Intake_g",
    "Fat_Intake_g",
    "Water_Intake_Liters"
]

invalid_report = []

for column in numeric_columns:

    if column not in df.columns:

        print(
            f"{column:<30}"
            f"COLUMN NOT FOUND"
        )

        continue

    numeric_data = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    non_numeric = int(
        numeric_data.isna().sum()
        - df[column].isna().sum()
    )

    non_positive = int(
        (numeric_data <= 0).sum()
    )

    total_invalid = (
        non_numeric + non_positive
    )

    invalid_report.append({
        "Parameter": column,
        "Non_Numeric": non_numeric,
        "Non_Positive": non_positive,
        "Total_Invalid": total_invalid
    })

    print(
        f"{column:<30}"
        f"Non-numeric: {non_numeric:<5}"
        f"Non-positive: {non_positive:<5}"
        f"Total: {total_invalid}"
    )


# 7. CATEGORICAL VALUE CHECK

print_heading(
    "7. CATEGORICAL PARAMETER CHECK"
)

categorical_columns = [
    "Gender",
    "Activity_Level",
    "Diet_Type",
    "Health_Status"
]

for column in categorical_columns:

    if column not in df.columns:

        print(
            f"\n{column}: COLUMN NOT FOUND"
        )

        continue

    print(f"\n{column}:")

    print(
        df[column].value_counts(
            dropna=False
        )
    )


# 8. OUTLIER CHECK

print_heading(
    "8. OUTLIER CHECK USING IQR"
)

outlier_report = []

for column in numeric_columns:

    if column not in df.columns:
        continue

    data = pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna()

    if len(data) == 0:
        continue

    Q1 = data.quantile(0.25)

    Q3 = data.quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = (
        Q1 - (1.5 * IQR)
    )

    upper_limit = (
        Q3 + (1.5 * IQR)
    )

    outliers = (
        (data < lower_limit)
        |
        (data > upper_limit)
    )

    outlier_count = int(
        outliers.sum()
    )

    outlier_report.append({
        "Parameter": column,
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "Lower_Limit": lower_limit,
        "Upper_Limit": upper_limit,
        "Outlier_Count": outlier_count
    })

    print(
        f"{column:<30}"
        f"Outliers: {outlier_count}"
    )


# 9. DIET TYPE GRAPH

if "Diet_Type" in df.columns:

    plt.figure(figsize=(8, 5))

    df["Diet_Type"].value_counts().plot(
        kind="bar",
        color="#4DB6AC",
        edgecolor="#2F6F68",
        linewidth=1.2
    )

    plt.title(
        "Diet Type Distribution"
    )

    plt.xlabel(
        "Diet Type"
    )

    plt.ylabel(
        "Number of Records"
    )

    plt.xticks(rotation=30)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            GRAPH_DIR,
            "diet_type_distribution.png"
        ),
        dpi=150
    )

    plt.close()


# 10. ACTIVITY LEVEL GRAPH

if "Activity_Level" in df.columns:

    plt.figure(figsize=(8, 5))

    df["Activity_Level"].value_counts().plot(
        kind="bar",
        color="#64B5F6",
        edgecolor="#3F7FAF",
        linewidth=1.2
    )

    plt.title(
        "Activity Level Distribution"
    )

    plt.xlabel(
        "Activity Level"
    )

    plt.ylabel(
        "Number of Records"
    )

    plt.xticks(rotation=30)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            GRAPH_DIR,
            "activity_level_distribution.png"
        ),
        dpi=150
    )

    plt.close()


# 11. HEALTH STATUS GRAPH

if "Health_Status" in df.columns:

    plt.figure(figsize=(8, 5))

    df["Health_Status"].value_counts().plot(
        kind="bar",
        color="#81C784",
        edgecolor="#4F8F52",
        linewidth=1.2
    )

    plt.title(
        "Health Status Distribution"
    )

    plt.xlabel(
        "Health Status"
    )

    plt.ylabel(
        "Number of Records"
    )

    plt.xticks(rotation=30)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            GRAPH_DIR,
            "health_status_distribution.png"
        ),
        dpi=150
    )

    plt.close()


# 12. AGE GRAPH

if "Age" in df.columns:

    plt.figure(figsize=(8, 5))

    plt.hist(
        pd.to_numeric(
            df["Age"],
            errors="coerce"
        ).dropna(),
        bins=20,
        color="#B39DDB",
        edgecolor="#6D5A91",
        linewidth=1.1
    )

    plt.title(
        "Age Distribution"
    )

    plt.xlabel(
        "Age"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            GRAPH_DIR,
            "age_distribution.png"
        ),
        dpi=150
    )

    plt.close()


# 13. BMI GRAPH

if "BMI" in df.columns:

    plt.figure(figsize=(8, 5))

    plt.hist(
        pd.to_numeric(
            df["BMI"],
            errors="coerce"
        ).dropna(),
        bins=20,
        color="#FFCC80",
        edgecolor="#B97832",
        linewidth=1.1
    )

    plt.title(
        "BMI Distribution"
    )

    plt.xlabel(
        "BMI"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            GRAPH_DIR,
            "bmi_distribution.png"
        ),
        dpi=150
    )

    plt.close()


# 14. CALORIE GRAPH

if "Daily_Calorie_Requirement" in df.columns:

    plt.figure(figsize=(8, 5))

    plt.hist(
        pd.to_numeric(
            df["Daily_Calorie_Requirement"],
            errors="coerce"
        ).dropna(),
        bins=20,
        color="#80DEEA",
        edgecolor="#428A92",
        linewidth=1.1
    )

    plt.title(
        "Daily Calorie Requirement Distribution"
    )

    plt.xlabel(
        "Daily Calories"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            GRAPH_DIR,
            "calorie_distribution.png"
        ),
        dpi=150
    )

    plt.close()

# 15. DATA CLEANING

print_heading(
    "15. DATA CLEANING"
)

original_records = len(df)


# Convert numeric columns

for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# Remove duplicate records

before_duplicates = len(df)

df = df.drop_duplicates()

removed_duplicates = (
    before_duplicates - len(df)
)

print(
    "Duplicate records removed:",
    removed_duplicates
)


# Remove missing records

before_missing = len(df)

df = df.dropna()

removed_missing = (
    before_missing - len(df)
)

print(
    "Records removed due to missing values:",
    removed_missing
)


# Remove invalid numeric records

before_invalid = len(df)

for column in numeric_columns:

    if column in df.columns:

        df = df[
            df[column] > 0
        ]

removed_invalid = (
    before_invalid - len(df)
)

print(
    "Records removed due to invalid values:",
    removed_invalid
)


print(
    "\nOriginal records:",
    original_records
)

print(
    "Records after cleaning:",
    len(df)
)

print(
    "Total records removed:",
    original_records - len(df)
)


# 16. DIET TYPES AFTER CLEANING

if "Diet_Type" in df.columns:

    print_heading(
        "16. DIET TYPES AFTER CLEANING"
    )

    diet_types = df[
        "Diet_Type"
    ].unique()

    print(
        "Total Diet Types:",
        len(diet_types)
    )

    for diet in diet_types:

        print(
            " -",
            diet
        )


# 17. ENCODING

print_heading(
    "17. ENCODING CATEGORICAL DATA"
)

gender_map = {
    "Male": 0,
    "Female": 1,
    "Other": 2
}

activity_map = {
    "Sedentary": 0,
    "Lightly Active": 1,
    "Moderately Active": 2,
    "Very Active": 3,
    "Athlete": 4
}

df["Gender_encoded"] = (
    df["Gender"].map(
        gender_map
    )
)

df["Activity_encoded"] = (
    df["Activity_Level"].map(
        activity_map
    )
)

df = df.dropna(
    subset=[
        "Gender_encoded",
        "Activity_encoded"
    ]
)


# 18. REGRESSION DATA

print_heading(
    "18. CALORIE PREDICTION - REGRESSION"
)

features = [
    "Age",
    "Gender_encoded",
    "Height_cm",
    "Weight_kg",
    "BMI",
    "Activity_encoded"
]

target = (
    "Daily_Calorie_Requirement"
)

X = df[features]

y = df[target]


# 19. TRAIN TEST SPLIT

print_heading(
    "19. TRAINING AND TESTING DATA"
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(
    "Training records:",
    len(X_train)
)

print(
    "Testing records:",
    len(X_test)
)


# 20. MULTIPLE LINEAR REGRESSION

print_heading(
    "20. MULTIPLE LINEAR REGRESSION"
)

print(
    "Training Multiple Linear Regression..."
)

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

linear_predictions = (
    linear_model.predict(
        X_test
    )
)

linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

linear_mse = mean_squared_error(
    y_test,
    linear_predictions
)

linear_rmse = np.sqrt(
    linear_mse
)

linear_r2 = r2_score(
    y_test,
    linear_predictions
)

print(
    f"MAE  : {linear_mae:.2f}"
)

print(
    f"MSE  : {linear_mse:.2f}"
)

print(
    f"RMSE : {linear_rmse:.2f}"
)

print(
    f"R²   : {linear_r2:.4f}"
)

print(
    f"R² % : {linear_r2 * 100:.2f}%"
)


# 21. RANDOM FOREST REGRESSION

print_heading(
    "21. RANDOM FOREST REGRESSION"
)

print(
    "Training Random Forest Regression..."
)

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)

rf_predictions = (
    rf_model.predict(
        X_test
    )
)

rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rf_mse = mean_squared_error(
    y_test,
    rf_predictions
)

rf_rmse = np.sqrt(
    rf_mse
)

rf_r2 = r2_score(
    y_test,
    rf_predictions
)

print(
    f"MAE  : {rf_mae:.2f}"
)

print(
    f"MSE  : {rf_mse:.2f}"
)

print(
    f"RMSE : {rf_rmse:.2f}"
)

print(
    f"R²   : {rf_r2:.4f}"
)

print(
    f"R² % : {rf_r2 * 100:.2f}%"
)


# 22. MODEL COMPARISON

print_heading(
    "22. MODEL COMPARISON"
)

comparison = pd.DataFrame({

    "Model": [
        "Multiple Linear Regression",
        "Random Forest Regression"
    ],

    "MAE": [
        linear_mae,
        rf_mae
    ],

    "MSE": [
        linear_mse,
        rf_mse
    ],

    "RMSE": [
        linear_rmse,
        rf_rmse
    ],

    "R2": [
        linear_r2,
        rf_r2
    ]
})

print(
    comparison.to_string(
        index=False
    )
)

comparison.to_csv(
    os.path.join(
        GRAPH_DIR,
        "regression_model_comparison.csv"
    ),
    index=False
)


# 23. ACTUAL VS PREDICTED GRAPH

print_heading(
    "23. ACTUAL VS PREDICTED GRAPH"
)

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    rf_predictions,
    color="#64B5F6",
    edgecolor="#3F7FAF",
    linewidth=0.8,
    alpha=0.75
)

plt.xlabel(
    "Actual Calories"
)

plt.ylabel(
    "Predicted Calories"
)

plt.title(
    "Actual vs Predicted Daily Calories - Random Forest"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        GRAPH_DIR,
        "actual_vs_predicted_calories.png"
    ),
    dpi=150
)

plt.close()

print(
    "Actual vs Predicted graph saved."
)

# 24. SAVE RANDOM FOREST MODEL

print_heading(
    "24. SAVING RANDOM FOREST MODEL"
)

model_data = {

    "model": rf_model,

    "gender_map": gender_map,

    "activity_map": activity_map,

    "features": features
}

joblib.dump(
    model_data,
    MODEL_PATH
)

print(
    "Random Forest model saved successfully:"
)

print(
    MODEL_PATH
)


# R² Score Comparison Chart
plt.figure(figsize=(8, 5))

plt.bar(
    comparison["Model"],
    comparison["R2"] * 100
)

plt.xlabel("Regression Algorithms")
plt.ylabel("R² Score (%)")
plt.title("Regression Algorithm Performance Comparison")
plt.ylim(0, 100)

plt.tight_layout()
plt.savefig(
    os.path.join(GRAPH_DIR, "r2_score_comparison.png"),
    dpi=150
)

plt.close()

print(
    "Regression Algorithm Performance Comparison graph saved."
)