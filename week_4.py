import pandas as pd
from pathlib import Path

file_path = Path("DataCoSupplyChainDataset.csv")

if file_path.exists():
    df = pd.read_csv(file_path, encoding="latin-1")

    print("Dataset loaded successfully!")
    print("Dataset shape:", df.shape)
    print("\nColumn names:")
    print(df.columns.tolist())

    print(df.head())
else:
    print("Dataset not found. Check the CSV filename.")

# 4.Inspect important columns

target = "Days for shipping (real)"

features = [
    "Days for shipment (scheduled)",
    "Benefit per order",
    "Sales per customer",
    "Order Item Quantity",
    "Order Item Product Price",
    "Shipping Mode",
    "Market",
    "Order Region"
]

print("Target variable:", target)
print("\nSelected features:")

for column in features:
    print(column, ":", df[column].dtype)

print("\nMissing values in selected columns:")

for column in features + [target]:
    print(column, ":", df[column].isnull().sum())

print("\nTarget variable statistics:")
print(df[target].describe())


# 5.Prepare data for machine learning

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define the target variable
target = "Days for shipping (real)"

# Select input features
numeric_features = [
    "Benefit per order",
    "Sales per customer",
    "Order Item Quantity",
    "Order Item Product Price"
]

categorical_features = [
    "Shipping Mode",
    "Market",
    "Order Region"
]

all_features = numeric_features + categorical_features

# Create input and target data
X = df[all_features]
y = df[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("Data preparation completed!")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Number of input features:", len(all_features))


# 6.Build and evaluate the model

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Process numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features)
    ]
)

# Create the predictive model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    ))
])

# Train the model
model.fit(X_train, y_train)

# Predict delivery duration
y_pred = model.predict(X_test)

# Evaluate performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Results")
print("------------------------")
print(f"MAE:  {mae:.4f} days")
print(f"RMSE: {rmse:.4f} days")
print(f"R-squared: {r2:.4f}")



# 7.Cross-validation

from sklearn.model_selection import KFold, cross_val_score

print("\nStarting cross-validation...")

# Use 3 folds to evaluate model consistency
cv = KFold(n_splits=3, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

# Convert negative scores to positive MAE values
cv_mae_scores = -cv_scores

print("\nCross-Validation Results")
print("------------------------")

for i, score in enumerate(cv_mae_scores, start=1):
    print(f"Fold {i} MAE: {score:.4f} days")

print(f"Average CV MAE: {cv_mae_scores.mean():.4f} days")
print(f"MAE Standard Deviation: {cv_mae_scores.std():.4f} days")


# Step 8: Logistics Optimization Analysis

import pandas as pd
import numpy as np

# Create a comparison between actual and predicted delivery times
results = pd.DataFrame({
    "Actual_Days": y_test,
    "Predicted_Days": y_pred,
    "Shipping_Mode": X_test["Shipping Mode"],
    "Market": X_test["Market"],
    "Order_Region": X_test["Order Region"]
})

results["Prediction_Error"] = (
    results["Actual_Days"] - results["Predicted_Days"]
).abs()

# Identify shipments with the largest prediction errors
print("\nTop 10 shipments with largest prediction errors:")
print(
    results.sort_values(
        "Prediction_Error", ascending=False
    ).head(10).to_string(index=False)
)

# Analyze performance by shipping mode
print("\nPrediction performance by shipping mode:")

mode_analysis = results.groupby("Shipping_Mode").agg(
    Shipment_Count=("Actual_Days", "size"),
    Average_Actual_Days=("Actual_Days", "mean"),
    Average_Predicted_Days=("Predicted_Days", "mean"),
    Average_Absolute_Error=("Prediction_Error", "mean")
).round(2)

print(mode_analysis.to_string())

# Analyze actual delivery time by region
print("\nAverage actual delivery time by region:")

region_analysis = results.groupby("Order_Region").agg(
    Shipment_Count=("Actual_Days", "size"),
    Average_Delivery_Days=("Actual_Days", "mean")
).sort_values("Average_Delivery_Days", ascending=False).round(2)

print(region_analysis.to_string())

# Save analysis outputs
results.to_csv("week4_prediction_results.csv", index=False)
mode_analysis.to_csv("week4_shipping_mode_analysis.csv")
region_analysis.to_csv("week4_regional_analysis.csv")

print("\nOptimization analysis completed.")
print("Three analysis CSV files have been saved.")



# Step 9: Generate data-based recommendations

print("\nLOGISTICS OPTIMIZATION RECOMMENDATIONS")
print("--------------------------------------")

slowest_region = region_analysis["Average_Delivery_Days"].idxmax()
slowest_region_days = region_analysis.loc[
    slowest_region, "Average_Delivery_Days"
]

highest_error_mode = mode_analysis["Average_Absolute_Error"].idxmax()
highest_error = mode_analysis.loc[
    highest_error_mode, "Average_Absolute_Error"
]

print(
    f"1. Review delivery operations in {slowest_region}, "
    f"where average actual delivery time is "
    f"{slowest_region_days:.2f} days in the test sample."
)

print(
    f"2. Review prediction reliability for {highest_error_mode}, "
    f"which has an average absolute prediction error of "
    f"{highest_error:.2f} days."
)

print(
    "3. Use predicted delivery durations to support planning, "
    "but validate proposed changes before implementation."
)

print(
    "4. Monitor delivery-time prediction errors regularly "
    "and retrain the model when performance declines."
)
