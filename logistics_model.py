"""
Week 4: Predictive Modeling and Optimization in Logistics Systems
All data is simulated for educational demonstration; it is not real operational data.
Run: pip install -r requirements.txt
     python logistics_model.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
df = pd.read_csv(ROOT / "logistics_predictive_dataset.csv")

target = "Delivery_Duration_min"
numeric_features = ["Distance_km", "Shipment_Weight_kg", "Shipment_Volume_units",
                    "Dispatch_Hour", "Transportation_Cost_INR"]
categorical_features = ["Transport_Mode", "Destination_Zone", "Traffic_Level", "Weather_Condition"]
X = df[numeric_features + categorical_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

def preprocessor():
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("numeric", num_pipe, numeric_features),
        ("categorical", cat_pipe, categorical_features)
    ])

candidates = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(
        random_state=42, max_depth=8, min_samples_leaf=5
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=180, random_state=42, min_samples_leaf=2, n_jobs=-1
    )
}
rows = []
for name, estimator in candidates.items():
    model = Pipeline([("prep", preprocessor()), ("model", estimator)])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rows.append({
        "Model": name,
        "MAE_min": mean_absolute_error(y_test, pred),
        "RMSE_min": mean_squared_error(y_test, pred) ** 0.5,
        "R2": r2_score(y_test, pred)
    })

# Tune Random Forest using 5-fold cross-validation and MAE as the search score.
rf = Pipeline([
    ("prep", preprocessor()),
    ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
])
grid = GridSearchCV(
    rf,
    {
        "model__n_estimators": [120, 180],
        "model__max_depth": [None, 14],
        "model__min_samples_leaf": [1, 3]
    },
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)
grid.fit(X_train, y_train)
best = grid.best_estimator_
pred = best.predict(X_test)
rows.append({
    "Model": "Tuned Random Forest",
    "MAE_min": mean_absolute_error(y_test, pred),
    "RMSE_min": mean_squared_error(y_test, pred) ** 0.5,
    "R2": r2_score(y_test, pred)
})
results = pd.DataFrame(rows).sort_values("RMSE_min")
results.to_csv(OUT / "model_performance.csv", index=False)
print(results.to_string(index=False))
print("Best parameters:", grid.best_params_)

# Predictive what-if scenario: reduce traffic level by one category.
# This is a scenario analysis, not proof that traffic can be controlled by the model.
X_scenario = X_test.copy()
X_scenario["Traffic_Level"] = X_scenario["Traffic_Level"].map(
    {"High": "Medium", "Medium": "Low", "Low": "Low"}
)
baseline = best.predict(X_test)
scenario = best.predict(X_scenario)
print(f"Mean baseline prediction: {baseline.mean():.1f} minutes")
print(f"Mean scenario prediction: {scenario.mean():.1f} minutes")
print(f"Estimated difference: {(baseline.mean()-scenario.mean()):.1f} minutes")

# Save actual-versus-predicted chart.
plt.figure(figsize=(7, 4.5))
plt.scatter(y_test, pred, alpha=0.55)
limits = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
plt.plot(limits, limits)
plt.xlabel("Actual duration (minutes)")
plt.ylabel("Predicted duration (minutes)")
plt.title("Actual vs Predicted Delivery Duration")
plt.tight_layout()
plt.savefig(OUT / "actual_vs_predicted.png", dpi=180)
