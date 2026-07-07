from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
#print( current_dir)
libs_path = os.path.join(current_dir, "..","..","_libs")
#print(libs_path)
sys.path.append(libs_path)

from select_features import FC_DimRed
from _new_ensemble_features_cv_selected import EnsembleClassifier
from utils import load_cov_mats, load_atms, upper_triangular_flatten

import warnings
import numpy as np
warnings.filterwarnings('ignore')

# TODO change here
for num_nodes in [20,35,50,78]:

    save_dir_base = os.path.join(current_dir, "..","..","..","Results_new",f"num_nodes_{num_nodes}")
    os.makedirs(save_dir_base, exist_ok=True)

    data_path = os.path.join(current_dir, "..","..","..","data","features") 


    # Load the data
    X_cov_mat, y_cov_mat = load_cov_mats(data_path)
    X_atm, y_atm = load_atms(data_path,zscore=1.6)
    # assert np.array_equal(y_cov_mat, y_atm), "Target labels for covariance matrices and ATMs do not match."

    # Select only the first 78x78 features
    X_cov_mat = X_cov_mat[:, :78, :78]
    X_atm = X_atm[:, :78, :78]  

    # Models definition
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model_cov_mat = LDA()
    model_atm = LDA()

    ensemble = EnsembleClassifier(model_cov_mat, model_atm)

    # TODO Change here (comment this part)
    # print("Dimension of Covariance Matrices before flattening: ", X_cov_mat.shape)
    # print("Dimension of ATMs before flattening: ", X_atm.shape)
    # X_cov_mat = np.array([upper_triangular_flatten(mat) for mat in X_cov_mat])
    # X_atm = np.array([upper_triangular_flatten(mat) for mat in X_atm])
    # print("Dimension of Covariance Matrices after flattening: ", X_cov_mat.shape)
    # print("Dimension of ATMs after flattening: ", X_atm.shape)

    # Fit the ensemble model
    # TODO change here
    ensemble.fit(X_cov_mat, X_atm, y_cov_mat, cv, threshold_feat_selection=0.1, num_nodes_feat_selection=num_nodes) 

    # Plot and save Results_new
    ensemble.plot_validation_metrics(output_path=save_dir_base+f"/figures/{model_cov_mat.__class__.__name__}/")
    ensemble.plot_oof_confusion_matrices(y_cov_mat, output_path=save_dir_base+f"/figures/{model_cov_mat.__class__.__name__}/")

    ensemble.save_metrics(output_path=save_dir_base+f"/logs/{model_cov_mat.__class__.__name__}/")

    ensemble.store_oof_probabilities(output_path=save_dir_base+f"/logs/{model_cov_mat.__class__.__name__}/")

    cv_scores = ensemble.get_cv_scores()
    print(f"Cross-validation scores: {cv_scores}")

    # Test predict function
    # TODO change here
    # predictions = ensemble.predict(X_cov_mat, X_atm)
    # print("Predictions: ", predictions)