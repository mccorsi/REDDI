from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from pyriemann.classification import TSClassifier
import os
import sys

sys.path.append(os.path.abspath("Code/_libs"))

from ensemble import EnsembleClassifier
from utils import load_cov_mats, load_atms
from select_features import FC_DimRed, AverageFilter

import warnings
import os
import json
warnings.filterwarnings('ignore')

# Load the data
X_cov_mat, y_cov_mat = load_cov_mats()
X_atm, y_atm = load_atms(zscore=1.6)

# Select only the first 78x78 features
X_cov_mat = X_cov_mat[:, :78, :78]
X_atm = X_atm[:, :78, :78]  

# Dimensionality reduction
dim_red_eta_50 = FC_DimRed(eta_threshold=0.1, nb_nodes=50)
# dim_red_eta_50 = AverageFilter(eta_threshold=0.1, nb_nodes=50)
X_cov_mat = dim_red_eta_50.fit_transform(X_cov_mat, y_cov_mat, metric='eta-squared')
print(f"Selected nodes Covariance Matrices: {dim_red_eta_50.node_select_}")

dim_red_eta_20 = FC_DimRed(eta_threshold=0.1, nb_nodes=20)
# dim_red_eta_20 = AverageFilter(eta_threshold=0.1, nb_nodes=20)
X_atm = dim_red_eta_20.fit_transform(X_atm, y_atm, metric='eta-squared')
print(f"Selected nodes ATMs: {dim_red_eta_20.node_select_}")

# Models definition
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model_cov_mat = TSClassifier(clf=KNeighborsClassifier(n_neighbors=5, metric='euclidean', weights='uniform'))  
# model_atm = TSClassifier(clf=KNeighborsClassifier(n_neighbors=3, metric='manhattan', weights='distance'))
model_atm = TSClassifier(clf=KNeighborsClassifier())

ensemble = EnsembleClassifier(model_cov_mat, model_atm)

# Fit the ensemble model
ensemble.fit(X_cov_mat, X_atm, y_cov_mat, cv) 

# Plot and save Results_new
ensemble.plot_validation_metrics(output_path=f"Results_new/figures/{model_cov_mat.__class__.__name__}/")
ensemble.plot_oof_confusion_matrices(y_cov_mat, output_path=f"Results_new/figures/{model_cov_mat.__class__.__name__}/")

ensemble.save_metrics(output_path=f"Results_new/logs/{model_cov_mat.__class__.__name__}/")

ensemble.store_oof_probabilities(output_path=f"Results_new/logs/{model_cov_mat.__class__.__name__}/")

cv_scores = ensemble.get_cv_scores()
print(f"Cross-validation scores: {cv_scores}")

# Test predict function
predictions = ensemble.predict(X_cov_mat, X_atm)
print("Predictions: ", predictions)