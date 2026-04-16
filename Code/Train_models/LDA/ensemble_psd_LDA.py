from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
#print( current_dir)
libs_path = os.path.join(current_dir, "..","..","_libs")
#print(libs_path)
sys.path.append(libs_path)

save_dir_base = os.path.join(current_dir, "..","..","..","Results_new")
os.makedirs(save_dir_base, exist_ok=True)

data_path = os.path.join(current_dir, "..","..","..","data","features")

from ensemble import EnsembleClassifier
from utils import load_psds


import warnings
warnings.filterwarnings('ignore')
save_path = "Results_new"

# Load the data
X_psd, y_psd = load_psds(data_path)

# Models definition
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model_psd_1 = LDA()
model_psd_2 = LDA() # this is not used, kept for compatibility with the ensemble code structure

ensemble = EnsembleClassifier(model_psd_1, model_psd_2, feature_names=['psd_1', 'psd_2'])

print("Dimension of PSD vector: ", X_psd.shape)

# Fit the ensemble model
ensemble.fit(X_psd, X_psd, y_psd, cv) 

# Plot and save Results_new
ensemble.plot_validation_metrics(output_path=f"{save_path}/figures/PSD/{model_psd_1.__class__.__name__}/")

ensemble.plot_oof_confusion_matrices(y_psd, output_path=f"{save_path}/figures/PSD/{model_psd_1.__class__.__name__}/")

ensemble.save_metrics(output_path=f"{save_path}/logs/PSD/{model_psd_1.__class__.__name__}/")

ensemble.store_oof_probabilities(output_path=f"{save_path}/logs/PSD/{model_psd_1.__class__.__name__}/")

cv_scores = ensemble.get_cv_scores()
print(f"Cross-validation scores: {cv_scores}")

# Test predict function
predictions = ensemble.predict(X_psd, X_psd)
print("Predictions: ", predictions)
