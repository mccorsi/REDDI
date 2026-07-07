from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from pyriemann.classification import TSClassifier
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
#print( current_dir)
libs_path = os.path.join(current_dir, "..","..","_libs")
#print(libs_path)
sys.path.append(libs_path)

data_path = os.path.join(current_dir, "..","..","..","data","features")

from _new_ensemble_features_cv_selected import EnsembleClassifier
from utils import load_cov_mats, load_atms
from select_features import FC_DimRed, AverageFilter

import warnings
import os
import json
warnings.filterwarnings('ignore')

for num_nodes in [20,35,50,78]:

    save_dir_base = os.path.join(current_dir, "..","..","..","Results_new", f"num_nodes_{num_nodes}")
    os.makedirs(save_dir_base, exist_ok=True)

    # Load the data
    X_cov_mat, y_cov_mat = load_cov_mats(data_path)
    X_atm, y_atm = load_atms(data_path, zscore=1.6)

    # Select only the first 78x78 features
    X_cov_mat = X_cov_mat[:, :78, :78]
    X_atm = X_atm[:, :78, :78]  

    # # Dimensionality reduction
    # dim_red_eta_50 = FC_DimRed(eta_threshold=0.1, nb_nodes=50)
    # # dim_red_eta_50 = AverageFilter(eta_threshold=0.1, nb_nodes=50)
    # X_cov_mat = dim_red_eta_50.fit_transform(X_cov_mat, y_cov_mat, metric='eta-squared')
    # print(f"Selected nodes Covariance Matrices: {dim_red_eta_50.node_select_}")

    # dim_red_eta_20 = FC_DimRed(eta_threshold=0.1, nb_nodes=20)
    # # dim_red_eta_20 = AverageFilter(eta_threshold=0.1, nb_nodes=20)
    # X_atm = dim_red_eta_20.fit_transform(X_atm, y_atm, metric='eta-squared')
    # print(f"Selected nodes ATMs: {dim_red_eta_20.node_select_}")

    # Models definition
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model_cov_mat = TSClassifier(clf=KNeighborsClassifier(n_neighbors=5, metric='euclidean', weights='uniform'))  
    # model_atm = TSClassifier(clf=KNeighborsClassifier(n_neighbors=3, metric='manhattan', weights='distance'))
    model_atm = TSClassifier(clf=KNeighborsClassifier())

    ensemble = EnsembleClassifier(model_cov_mat, model_atm)

    # Fit the ensemble model
    ensemble.fit(X_cov_mat, X_atm, y_cov_mat, cv, threshold_feat_selection=0.1, num_nodes_feat_selection=num_nodes, riemanian_classifier=True)  

    # Plot and save Results_new
    ensemble.plot_validation_metrics(output_path=save_dir_base+f"/figures/{model_cov_mat.__class__.__name__}/")
    ensemble.plot_oof_confusion_matrices(y_cov_mat, output_path=save_dir_base+f"/figures/{model_cov_mat.__class__.__name__}/")

    ensemble.save_metrics(output_path=save_dir_base+f"/logs/{model_cov_mat.__class__.__name__}/")

    ensemble.store_oof_probabilities(output_path=save_dir_base+f"/logs/{model_cov_mat.__class__.__name__}/")

    cv_scores = ensemble.get_cv_scores()
    print(f"Cross-validation scores: {cv_scores}")

    # Test predict function
    # predictions = ensemble.predict(X_cov_mat, X_atm)
    # print("Predictions: ", predictions)