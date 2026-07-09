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

from ensemble import EnsembleClassifier
from utils import load_atms, load_corr_mats
from select_features import FC_DimRed

import warnings
warnings.filterwarnings('ignore')


for num_nodes in [20,35,50,78]:

    save_dir_base = os.path.join(current_dir, "..","..","..","Results_new", f"num_nodes_{num_nodes}")
    os.makedirs(save_dir_base, exist_ok=True)

    # Load the data
    X_corr_mat, y_corr_mat = load_corr_mats(data_path)
    X_atm, y_atm = load_atms(data_path, zscore=1.6)

    # Select only the first 78x78 features
    X_corr_mat = X_corr_mat[:, :78, :78]  
    X_atm_mat = X_atm[:, :78, :78]

    # Models definition
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model_corr_mat = TSClassifier(clf=KNeighborsClassifier())
    model_atm = TSClassifier(clf=KNeighborsClassifier())

    ensemble = EnsembleClassifier(model_corr_mat, model_atm, feature_names=['correlation_matrices', 'atms'])

    # Fit the ensemble model
    ensemble.fit(X_corr_mat, X_atm_mat, y_corr_mat, cv, threshold_feat_selection=0.1, num_nodes_feat_selection=num_nodes, riemanian_classifier=True)  

    # Plot and save Results_new
    ensemble.plot_validation_metrics(output_path=save_dir_base+f"/figures/output_Corr_ATM_ensamble/riemann/")
    ensemble.plot_oof_confusion_matrices(y_corr_mat, output_path=save_dir_base+f"/figures/output_Corr_ATM_ensamble/riemann/")

    ensemble.save_metrics(output_path=save_dir_base+f"/logs/output_Corr_ATM_ensamble/riemann/")

    ensemble.store_oof_probabilities(output_path=save_dir_base+f"/logs/output_Corr_ATM_ensamble/riemann/")

    cv_scores = ensemble.get_cv_scores()
    print(f"Cross-validation scores: {cv_scores}")

    # Test predict function
    # predictions = ensemble.predict(X_corr_mat, X_atm_mat)
    # print("Predictions: ", predictions)