from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import os
import sys

sys.path.append(os.path.abspath("Code/_libs"))

from ensemble import EnsembleClassifier
from utils import load_cov_mats, load_corr_mats, upper_triangular_flatten
from select_features import FC_DimRed

import warnings
import numpy as np
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

model_cov_mat = SVC(probability=True)
model_corr_mat = SVC(probability=True)

ensemble = EnsembleClassifier(model_cov_mat, model_corr_mat, feature_names=['covariance_matrices', 'correlation_matrices'])

print("Dimension of Covariance Matrices before flattening: ", X_cov_mat.shape)
print("Dimension of Correlation Matrices before flattening: ", X_corr_mat.shape)
X_cov_mat = np.array([upper_triangular_flatten(mat) for mat in X_cov_mat])
X_corr_mat = np.array([upper_triangular_flatten(mat) for mat in X_corr_mat])
print("Dimension of Covariance Matrices after flattening: ", X_cov_mat.shape)
print("Dimension of Correlation Matrices after flattening: ", X_corr_mat.shape)

# Fit the ensemble model
ensemble.fit(X_cov_mat, X_corr_mat, y_cov_mat, cv) 

# Plot and save results
ensemble.plot_validation_metrics(output_path=f"Results/figures/Corr_Mat/{model_cov_mat.__class__.__name__}/")
ensemble.plot_oof_confusion_matrices(y_cov_mat, output_path=f"Results/figures/Corr_Mat/{model_cov_mat.__class__.__name__}/")

ensemble.save_metrics(output_path=f"Results/logs/Corr_Mat/{model_cov_mat.__class__.__name__}/")

ensemble.store_oof_probabilities(output_path=f"Results/logs/Corr_Mat/{model_cov_mat.__class__.__name__}/")

cv_scores = ensemble.get_cv_scores()
print(f"Cross-validation scores: {cv_scores}")

# Test predict function
predictions = ensemble.predict(X_cov_mat, X_corr_mat)
print("Predictions: ", predictions)