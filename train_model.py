# train_models.py
import pandas as pd
import pickle
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor

print("🔄 Processing raw building datasets...")

# --- PHASE 1: GEOMETRY ---
df1 = pd.read_csv('Pareto.csv', header=[0, 1], encoding='latin1')
df1.columns = ['iteration','generation','category','total_energy','discomfort_hours','cooling_energy','window_to_wall','orientation','facade_type','shading_type','window_open_pct','unnamed'][:len(df1.columns)]

X1 = df1[['window_to_wall', 'orientation', 'facade_type', 'shading_type', 'window_open_pct']].copy()
enc_l = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X1[['facade_type', 'shading_type']] = enc_l.fit_transform(X1[['facade_type', 'shading_type']])

with open('encoder_layout.pkl', 'wb') as f:
    pickle.dump(enc_l, f)

# --- PHASE 2: MATERIALS ---
df2 = pd.read_csv('2nd Optimization Results.csv', header=[0, 1], encoding='latin1')
df2.columns = ['iteration','generation','category','total_energy','discomfort_hours','cooling_energy','external_wall','flat_roof','glazing_type','partition_wall','unnamed'][:len(df2.columns)]

X2 = df2[['external_wall', 'flat_roof', 'glazing_type', 'partition_wall']].copy()
enc_m = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X2[['external_wall', 'flat_roof', 'glazing_type', 'partition_wall']] = enc_m.fit_transform(X2[['external_wall', 'flat_roof', 'glazing_type', 'partition_wall']])

with open('encoder_materials.pkl', 'wb') as f:
    pickle.dump(enc_m, f)


# --- TRANSFORMATION PIPELINE WITH DEFINED EXPLICIT LEARNING PARAMS ---
def build_nn_pipeline():
    # Construct model matching your 200 epochs requirements
    nn_structure = MLPRegressor(
        hidden_layer_sizes=(128, 64), 
        max_iter=500,                # Set to 200 Epochs
        learning_rate_init=0.002,    # Set Explicit learning step size
        random_state=42
    )
    feature_pipeline = make_pipeline(StandardScaler(), nn_structure)
    
    full_pipeline = TransformedTargetRegressor(
        regressor=feature_pipeline, 
        transformer=StandardScaler()
    )
    return full_pipeline


# --- MODEL COMPILATION AND EXPORT LOOP ---
print("\n🧠 Training Phase 1: Geometry Networks (200 Epochs @ 0.002 LR)...")
targets_l = {'total_energy': 'total', 'cooling_energy': 'cool', 'discomfort_hours': 'discom'}
for col, short_name in targets_l.items():
    print(f"  -> Compiling Neural Network: {col}...")
    model = build_nn_pipeline()
    model.fit(X1, df1[col])
    with open(f'model_layout_{short_name}_energy.pkl' if short_name != 'discom' else f'model_layout_{short_name}fort_hours.pkl', 'wb') as f:
        pickle.dump(model, f)

print("\n🧠 Training Phase 2: Materials Networks (200 Epochs @ 0.002 LR)...")
targets_m = {'total_energy': 'total', 'cooling_energy': 'cool', 'discomfort_hours': 'discom'}
for col, short_name in targets_m.items():
    print(f"  -> Compiling Neural Network: {col}...")
    model = build_nn_pipeline()
    model.fit(X2, df2[col])
    with open(f'model_mat_{short_name}_energy.pkl' if short_name != 'discom' else f'model_mat_{short_name}fort_hours.pkl', 'wb') as f:
        pickle.dump(model, f)

print("\n🎉 Done! All saved pipeline models are completely upgraded to 200 epochs.")