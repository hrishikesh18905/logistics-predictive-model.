# Week 4: Predictive Modeling and Optimization in Logistics Systems

## Project overview
This project predicts shipment delivery duration in minutes using simulated logistics data. It compares Linear Regression, Decision Tree Regression, Random Forest Regression, and a tuned Random Forest model. It also demonstrates a what-if scenario for reducing traffic severity.

**Important:** All rows are hypothetical and generated for learning. They are not real business records or a production-ready model.

## Files
- `logistics_predictive_dataset.csv` — 1,200 simulated shipment rows.
- `logistics_model.py` — preprocessing, model training, evaluation, cross-validation tuning, and scenario analysis.
- `requirements.txt` — required Python packages.
- `outputs/` — charts and metric summaries used in the report.

## Run locally
1. Install Python 3.10 or newer.
2. Open this folder in VS Code.
3. In the terminal, run:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   pip install -r requirements.txt
   python logistics_model.py
   ```

## Target and features
Target: `Delivery_Duration_min`

Features: distance, weight, volume, dispatch hour, transportation cost, transport mode, destination zone, traffic level, and weather condition.

## Evaluation
The script reports MAE, RMSE, and R-squared on a held-out 20% test set. Random Forest hyperparameters are tuned using 5-fold shuffled cross-validation on the training set.

## GitHub submission
Create a new **public** GitHub repository (unless your internship allows private repositories), for example:
`yuva-intern-week4-logistics-predictive-model`

Upload the contents of this project folder, commit the files, and copy the repository URL. Do not submit a placeholder URL; use the real URL shown on your GitHub repository page.
