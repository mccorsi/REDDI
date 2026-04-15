from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from pyriemann.classification import TSClassifier
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
from utils import load_cov_mats, load_corr_mats
from select_features import FC_DimRed

import warnings
import os
import json
warnings.filterwarnings('ignore')

# Load the data
X_cov_mat, y_cov_mat = load_cov_mats()
X_corr_mat, y_corr_mat = load_corr_mats()

# Select only the first 78x78 features
X_cov_mat = X_cov_mat[:, :78, :78]
X_corr_mat = X_corr_mat[:, :78, :78]  

# Dimensionality reduction
dim_red_eta_50 = FC_DimRed(eta_threshold=0.1, nb_nodes=50)
X_cov_mat = dim_red_eta_50.fit_transform(X_cov_mat, y_cov_mat, metric='eta-squared')
print(f"Selected nodes Covariance Matrices: {dim_red_eta_50.node_select_}")

dim_red_eta_20 = FC_DimRed(eta_threshold=0.1, nb_nodes=20)
X_corr_mat = dim_red_eta_20.fit_transform(X_corr_mat, y_corr_mat, metric='eta-squared')
print(f"Selected nodes Correlation Matrices: {dim_red_eta_20.node_select_}")

# Models definition
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model_cov_mat = TSClassifier(clf=KNeighborsClassifier(n_neighbors=5, metric='euclidean', weights='uniform'))  
model_corr_mat = TSClassifier(clf=KNeighborsClassifier())

ensemble = EnsembleClassifier(model_cov_mat, model_corr_mat, feature_names=['covariance_matrices', 'correlation_matrices'])

# Fit the ensemble model
ensemble.fit(X_cov_mat, X_corr_mat, y_cov_mat, cv) 

# Plot and save Results_new
ensemble.plot_validation_metrics(output_path=save_dir_base+f"/figures/Corr_Mat/{model_cov_mat.__class__.__name__}/")
ensemble.plot_oof_confusion_matrices(y_cov_mat, output_path=save_dir_base+f"/figures/Corr_Mat/{model_cov_mat.__class__.__name__}/")

ensemble.save_metrics(output_path=save_dir_base+f"/logs/Corr_Mat/{model_cov_mat.__class__.__name__}/")

ensemble.store_oof_probabilities(output_path=save_dir_base+f"/logs/Corr_Mat/{model_cov_mat.__class__.__name__}/")

cv_scores = ensemble.get_cv_scores()
print(f"Cross-validation scores: {cv_scores}")

# Test predict function
predictions = ensemble.predict(X_cov_mat, X_corr_mat)
print("Predictions: ", predictions)