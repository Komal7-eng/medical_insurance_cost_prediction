# for data manipulation
# for data manipulation
import pandas as pd
import numpy as np

# preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# model training
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# saving model
import joblib
import os

# hugging face
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
from dotenv import load_dotenv

# 🔥 LOAD TOKEN
load_dotenv()

# 🔥 INIT API
api = HfApi(token=os.getenv("HF_TOKEN"))

# ---------------- DATA ---------------- #
Xtrain = pd.read_csv("hf://datasets/kaushalya7/medical-insurance-cost-prediction/Xtrain.csv")
Xtest = pd.read_csv("hf://datasets/kaushalya7/medical-insurance-cost-prediction/Xtest.csv")
ytrain = pd.read_csv("hf://datasets/kaushalya7/medical-insurance-cost-prediction/ytrain.csv")
ytest = pd.read_csv("hf://datasets/kaushalya7/medical-insurance-cost-prediction/ytest.csv")

# 🔥 FIX SHAPE (IMPORTANT)
ytrain = ytrain.values.ravel()
ytest = ytest.values.ravel()

# 🔥 CHECK TOKEN
if os.getenv("HF_TOKEN") is None:
    raise ValueError("❌ HF_TOKEN not found")

# 🔥 VERIFY USER
print("User:", api.whoami())

# ---------------- FEATURES ---------------- #
numeric_features = ['age', 'bmi', 'children']
categorical_features = ['sex', 'smoker', 'region']

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# ---------------- MODEL ---------------- #
xgb_model = xgb.XGBRegressor(random_state=42, objective="reg:squarederror")

param_grid = {
    'xgbregressor__n_estimators': [50, 100],
    'xgbregressor__max_depth': [2, 3],
    'xgbregressor__learning_rate': [0.01, 0.05],
}

pipeline = make_pipeline(preprocessor, xgb_model)

grid_search = GridSearchCV(
    pipeline, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1
)

grid_search.fit(Xtrain, ytrain)

best_model = grid_search.best_estimator_

print("Best Params:", grid_search.best_params_)

# ---------------- EVALUATION ---------------- #
y_pred = best_model.predict(Xtest)

print("MAE:", mean_absolute_error(ytest, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(ytest, y_pred)))
print("R2:", r2_score(ytest, y_pred))

# ---------------- SAVE MODEL ---------------- #
model_path = "best_medical_insurance_model_v1.joblib"
joblib.dump(best_model, model_path)

print("✅ Model saved locally")

# ---------------- UPLOAD TO HF ---------------- #
repo_id = "kaushalya7/medical_insurance_model"
repo_type = "model"

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"✅ Model repo '{repo_id}' already exists")
except RepositoryNotFoundError:
    print(f"🚀 Creating model repo '{repo_id}'...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)

# upload
try:
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print("🎉 Model uploaded successfully!")
except Exception as e:
    print("❌ Upload failed:", e)

# 🔍 verify
print("Files in repo:", api.list_repo_files(repo_id=repo_id, repo_type="model"))
