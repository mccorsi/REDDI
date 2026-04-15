from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
print( current_dir)
libs_path = os.path.join(current_dir, "..","..","_libs")
print(libs_path)
sys.path.append(libs_path)

save_dir_base = os.path.join(current_dir, "..","..","..","Results_new")
os.makedirs(save_dir_base, exist_ok=True)

from ensemble import EnsembleClassifier
from utils import load_cov_mats, load_atms, upper_triangular_flatten
from select_features import FC_DimRed

import warnings
import numpy as np
warnings.filterwarnings('ignore')

# Load the data
X_cov_mat, y_cov_mat = load_cov_mats()
X_atm, y_atm = load_atms(zscore=1.6)

# Select only the first 78x78 features
X_cov_mat = X_cov_mat[:, :78, :78]
X_atm = X_atm[:, :78, :78]  

# Dimensionality reduction
dim_red_eta_50 = FC_DimRed(eta_threshold=0.1, nb_nodes=50)
X_cov_mat = dim_red_eta_50.fit_transform(X_cov_mat, y_cov_mat, metric='eta-squared')
print(f"Selected nodes Covariance Matrices: {dim_red_eta_50.node_select_}")

dim_red_eta_20 = FC_DimRed(eta_threshold=0.1, nb_nodes=20)
X_atm = dim_red_eta_20.fit_transform(X_atm, y_atm, metric='eta-squared')
print(f"Selected nodes ATMs: {dim_red_eta_20.node_select_}")

# Models definition
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model_cov_mat = SVC(kernel='rbf', probability=True, random_state=42)
model_atm = SVC(kernel='rbf', probability=True, random_state=42)

ensemble = EnsembleClassifier(model_cov_mat, model_atm)

print("Dimension of Covariance Matrices before flattening: ", X_cov_mat.shape)
print("Dimension of ATMs before flattening: ", X_atm.shape)
X_cov_mat = np.array([upper_triangular_flatten(mat) for mat in X_cov_mat])
X_atm = np.array([upper_triangular_flatten(mat) for mat in X_atm])
print("Dimension of Covariance Matrices after flattening: ", X_cov_mat.shape)
print("Dimension of ATMs after flattening: ", X_atm.shape)

# Fit the ensemble model
ensemble.fit(X_cov_mat, X_atm, y_cov_mat, cv) 

# Plot and save Results_new
ensemble.plot_validation_metrics(output_path=save_dir_base+f"/figures/{model_cov_mat.__class__.__name__}/")
ensemble.plot_oof_confusion_matrices(y_cov_mat, output_path=save_dir_base+f"/figures/{model_cov_mat.__class__.__name__}/")

ensemble.save_metrics(output_path=save_dir_base+f"/logs/{model_cov_mat.__class__.__name__}/")

ensemble.store_oof_probabilities(output_path=save_dir_base+f"/logs/{model_cov_mat.__class__.__name__}/")

cv_scores = ensemble.get_cv_scores()
print(f"Cross-validation scores: {cv_scores}")

# Test predict function
predictions = ensemble.predict(X_cov_mat, X_atm)
print("Predictions: ", predictions)