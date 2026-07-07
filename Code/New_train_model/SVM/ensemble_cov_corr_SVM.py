from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
#print( current_dir)
libs_path = os.path.join(current_dir, "..","..","_libs")
#print(libs_path)
sys.path.append(libs_path)

# TODO qui
data_path = os.path.join(current_dir, "..","..","..","data","features")

from _new_ensemble_features_cv_selected import EnsembleClassifier # TODO qui
from utils import load_cov_mats, load_corr_mats, upper_triangular_flatten
from select_features import FC_DimRed

import warnings
import numpy as np
warnings.filterwarnings('ignore')

for num_nodes in [20,35,50,78]: #

    save_dir_base = os.path.join(current_dir, "..","..","..","Results_new", f"num_nodes_{num_nodes}")
    os.makedirs(save_dir_base, exist_ok=True)

    # Load the data
    X_cov_mat, y_cov_mat = load_cov_mats(data_path)
    X_corr_mat, y_corr_mat = load_corr_mats(data_path)

    # Select only the first 78x78 features
    X_cov_mat = X_cov_mat[:, :78, :78]
    X_corr_mat = X_corr_mat[:, :78, :78]  

    # Models definition
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model_cov_mat = SVC(probability=True)
    model_corr_mat = SVC(probability=True)

    ensemble = EnsembleClassifier(model_cov_mat, model_corr_mat, feature_names=['covariance_matrices', 'correlation_matrices'])

    # Fit the ensemble model #
    ensemble.fit(X_cov_mat, X_corr_mat, y_cov_mat, cv, threshold_feat_selection=0.1, num_nodes_feat_selection=num_nodes) 

    # Plot and save Results_new
    ensemble.plot_validation_metrics(output_path=save_dir_base+f"/figures/Corr_Mat/{model_cov_mat.__class__.__name__}/")
    ensemble.plot_oof_confusion_matrices(y_cov_mat, output_path=save_dir_base+f"/figures/Corr_Mat/{model_cov_mat.__class__.__name__}/")

    ensemble.save_metrics(output_path=save_dir_base+f"/logs/Corr_Mat/{model_cov_mat.__class__.__name__}/")

    ensemble.store_oof_probabilities(output_path=save_dir_base+f"/logs/Corr_Mat/{model_cov_mat.__class__.__name__}/")

    cv_scores = ensemble.get_cv_scores()
    print(f"Cross-validation scores: {cv_scores}")
