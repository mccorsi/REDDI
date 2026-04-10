from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import os
import sys

sys.path.append(os.path.abspath("Code/_libs"))

from ensemble import EnsembleClassifier
from utils import load_psds

import warnings
import numpy as np
warnings.filterwarnings('ignore')

# Load the data
X_psd, y_psd = load_psds()

# Models definition
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model_psd_1 = SVC(kernel='rbf', probability=True, random_state=42)
model_psd_2 = SVC(kernel='rbf', probability=True, random_state=42) # this is not used, kept for compatibility with the ensemble code structure

ensemble = EnsembleClassifier(model_psd_1, model_psd_2, feature_names=['psd_1', 'psd_2'])

print("Dimension of PSD vector: ", X_psd.shape)

# Fit the ensemble model
ensemble.fit(X_psd, X_psd, y_psd, cv) 

# Plot and save Results_new
ensemble.plot_validation_metrics(output_path=f"Results_new/figures/PSD/{model_psd_1.__class__.__name__}/")

ensemble.plot_oof_confusion_matrices(y_psd, output_path=f"Results_new/figures/PSD/{model_psd_1.__class__.__name__}/")

ensemble.save_metrics(output_path=f"Results_new/logs/PSD/{model_psd_1.__class__.__name__}/")

ensemble.store_oof_probabilities(output_path=f"Results_new/logs/PSD/{model_psd_1.__class__.__name__}/")

cv_scores = ensemble.get_cv_scores()
print(f"Cross-validation scores: {cv_scores}")

# Test predict function
predictions = ensemble.predict(X_psd, X_psd)
print("Predictions: ", predictions)
