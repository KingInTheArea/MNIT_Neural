# verify_nn_epochs.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

scaler_X = StandardScaler()

#1st pahse

print("⏳ Verifying Phase 1 (Geometry)...")
df1 = pd.read_csv('Pareto.csv', header=[0, 1], encoding='latin1')
df1.columns = ['iteration','generation','category','total_energy','discomfort_hours','cooling_energy','window_to_wall','orientation','facade_type','shading_type','window_open_pct','unnamed'][:len(df1.columns)]

X1 = df1[['window_to_wall', 'orientation', 'facade_type', 'shading_type', 'window_open_pct']].copy()
X1[['facade_type', 'shading_type']] = OrdinalEncoder().fit_transform(X1[['facade_type', 'shading_type']])
y1 = df1['total_energy']

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)

X_train1_scaled = scaler_X.fit_transform(X_train1)
X_test1_scaled = scaler_X.transform(X_test1)

y1_mean, y1_std = y_train1.mean(), y_train1.std()
y_train1_scaled = (y_train1 - y1_mean) / y1_std


nn_model1 = MLPRegressor(
    hidden_layer_sizes=(128, 64), 
    max_iter=500,                # 200 Epochs
    learning_rate_init=0.002,    # Learning Rate 
    random_state=42
)
nn_model1.fit(X_train1_scaled, y_train1_scaled)

preds_scaled1 = nn_model1.predict(X_test1_scaled)
preds1 = (preds_scaled1 * y1_std) + y1_mean

print("\n--- PHASE 1: NEURAL NETWORK VERIFICATION (200 TARGETS) ---")
print(f"Learning Rate Used: {nn_model1.learning_rate_init}")
print(f"Accuracy Score (R²): {r2_score(y_test1, preds1)*100:.2f}%")
print(f"Average Prediction Deviation (MAE): {mean_absolute_error(y_test1, preds1):.2f} kWh")


#2nd pahse
print("\n⏳ Verifying Phase 2 (Materials)...")
df2 = pd.read_csv('2nd Optimization Results.csv', header=[0, 1], encoding='latin1')
df2.columns = ['iteration','generation','category','total_energy','discomfort_hours','cooling_energy','external_wall','flat_roof','glazing_type','partition_wall','unnamed'][:len(df2.columns)]

X2 = df2[['external_wall', 'flat_roof', 'glazing_type', 'partition_wall']].copy()
X2[['external_wall', 'flat_roof', 'glazing_type', 'partition_wall']] = OrdinalEncoder().fit_transform(X2[['external_wall', 'flat_roof', 'glazing_type', 'partition_wall']])
y2 = df2['total_energy']

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)

X_train2_scaled = scaler_X.fit_transform(X_train2)
X_test2_scaled = scaler_X.transform(X_test2)

y2_mean, y2_std = y_train2.mean(), y_train2.std()
y_train2_scaled = (y_train2 - y2_mean) / y2_std

# Configured for 200 epochs and explicit learning rate
nn_model2 = MLPRegressor(
    hidden_layer_sizes=(128, 64), 
    max_iter=500,                # 200 Epochs
    learning_rate_init=0.002,    # Learning Rate Controlled Here
    random_state=42
)
nn_model2.fit(X_train2_scaled, y_train2_scaled)

preds_scaled2 = nn_model2.predict(X_test2_scaled)
preds2 = (preds_scaled2 * y2_std) + y2_mean

print("\n--- PHASE 2: NEURAL NETWORK VERIFICATION (200 TARGETS) ---")
print(f"Learning Rate Used: {nn_model2.learning_rate_init}")
print(f"Accuracy Score (R²): {r2_score(y_test2, preds2)*100:.2f}%")
print(f"Average Prediction Deviation (MAE): {mean_absolute_error(y_test2, preds2):.2f} kWh")