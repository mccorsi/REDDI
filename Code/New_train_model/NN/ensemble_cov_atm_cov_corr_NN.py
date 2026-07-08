from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import os
import sys
import numpy as np
import warnings
import json
warnings.filterwarnings('ignore')

current_dir = os.path.dirname(os.path.abspath(__file__))
#print( current_dir)
libs_path = os.path.join(current_dir, "..", "..", "_libs")
#print(libs_path)
sys.path.append(libs_path)

from _new_ensemble_features_cv_selected import EnsembleClassifier
from utils import load_cov_mats, load_atms, load_corr_mats


# ===============================================================
# Neural network architectures
# ===============================================================
def build_simple_nn(input_dim, num_classes):
    model = Sequential([
        Dense(128, activation="relu", input_dim=input_dim),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_deep_nn(input_dim, num_classes):
    model = Sequential([
        Dense(256, activation="relu", input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ===============================================================
# Adapter: Keras models become usable as model1 / model2
# of EnsembleClassifier. Adapted to .fit(X, y) and .predict_proba(X).
# ===============================================================
class KerasEnsembleModel:
    build_fn = None    # impostata dalla sottoclasse
    epochs = 50        # impostata dalla sottoclasse
    batch_size = 16
    verbose = 0

    def __init__(self):
        self.model_ = None

    def fit(self, X, y):
        num_classes = len(np.unique(y))
        self.model_ = self.build_fn(X.shape[1], num_classes)
        self.model_.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        return self

    def predict_proba(self, X):
        return self.model_.predict(X, verbose=self.verbose)


class SimpleNN(KerasEnsembleModel):
    build_fn = staticmethod(build_simple_nn)
    epochs = 50


class DeepNN(KerasEnsembleModel):
    build_fn = staticmethod(build_deep_nn)
    epochs = 80


NN_ARCHITECTURES = {
    "SimpleNN": SimpleNN,
    "DeepNN": DeepNN,
}


# ===============================================================
# Load my Data
# ===============================================================
data_path = os.path.join(current_dir, "..", "..", "..", "data", "features")

X_cov_mat, y_cov_mat = load_cov_mats(data_path)
X_atm, y_atm = load_atms(data_path, zscore=1.6)
X_corr, y_corr = load_corr_mats(data_path)
assert np.array_equal(y_cov_mat, y_atm), "Target labels for covariance matrices and ATMs do not match."
assert np.array_equal(y_cov_mat, y_corr), "Target labels for covariance matrices and correlation matrices do not match."

# Select only the first 78x78 features
X_cov_mat = X_cov_mat[:, :78, :78]
X_atm = X_atm[:, :78, :78]
X_corr = X_corr[:, :78, :78]

# Keras richiede label intere per sparse_categorical_crossentropy
# (no-op se y_cov_mat e' gia' intera)
le = LabelEncoder()
y_cov_mat = le.fit_transform(y_cov_mat)

ensemble_pairs = {
    "COV+ATM":  dict(X_1=X_cov_mat, X_2=X_atm,  feature_names=['covariance_matrices', 'atms']),
    "COV+CORR": dict(X_1=X_cov_mat, X_2=X_corr, feature_names=['covariance_matrices', 'correlation_matrices']),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ===============================================================
# TODO change here
for num_nodes in [20, 35, 50,78]:

    save_dir_base = os.path.join(current_dir, "..", "..", "..", "Results_new", f"num_nodes_{num_nodes}")
    os.makedirs(save_dir_base, exist_ok=True)

    for pair_name, pair_cfg in ensemble_pairs.items():
        for arch_name, ArchClass in NN_ARCHITECTURES.items():

            print(f"\n===== num_nodes={num_nodes} | {pair_name} | {arch_name} =====")

            # Due istanze SEPARATE: EnsembleClassifier tiene model1/model2
            # per tutta la CV, quindi non devono essere lo stesso oggetto
            # (altrimenti il secondo .fit() sovrascriverebbe i pesi del primo).
            model_1 = ArchClass()
            model_2 = ArchClass()

            ensemble = EnsembleClassifier(model_1, model_2, feature_names=pair_cfg["feature_names"])

            ensemble.fit(
                pair_cfg["X_1"], pair_cfg["X_2"], y_cov_mat, cv,
                threshold_feat_selection=0.1, num_nodes_feat_selection=num_nodes
            )

            out_tag = f"{pair_name}/{arch_name}"
            ensemble.plot_validation_metrics(output_path=save_dir_base + f"/figures/{out_tag}/")
            ensemble.plot_oof_confusion_matrices(y_cov_mat, output_path=save_dir_base + f"/figures/{out_tag}/")
            ensemble.save_metrics(output_path=save_dir_base + f"/logs/{out_tag}/")
            ensemble.store_oof_probabilities(output_path=save_dir_base + f"/logs/{out_tag}/")

            cv_scores = ensemble.get_cv_scores()
            print(f"[{num_nodes} | {pair_name} | {arch_name}] Cross-validation scores: {cv_scores}")


# =====================================================================
# Convert Results in an appropriate format for the benchmark plots
# =====================================================================
# Folder containing the current (old-format) results
old_results = "Results_new"

# Folder where the standardized results will be saved
new_results = "Results_new"

os.makedirs(new_results, exist_ok=True)

# ------------------------------------------------------
# Search recursively for SimpleNN and DeepNN folders
# ------------------------------------------------------
for root, dirs, files in os.walk(old_results):

    # Process ONLY folders named SimpleNN or DeepNN
    model_name = os.path.basename(root)
    if model_name not in ["SimpleNN", "DeepNN"]:
        continue

    # Skip if metrics.json does not exist
    if "metrics.json" not in files:
        #print(f"Skipping {root}: metrics.json not found.")
        continue

    metrics_file = os.path.join(root, "metrics.json")

    # ------------------------------------------------------
    # Load metrics
    # ------------------------------------------------------
    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    train_metrics = metrics["train_metrics_ensemble"]
    val_metrics = metrics["validation_metrics_ensemble"]

    # ------------------------------------------------------
    # Extract arrays (same format as sklearn.cross_validate)
    # ------------------------------------------------------
    bal_acc_train = np.array([fold["balanced_accuracy"] for fold in train_metrics])

    bal_acc_val = np.array([fold["balanced_accuracy"] for fold in val_metrics])

    f1_train = np.array([fold["f1_macro"] for fold in train_metrics])

    f1_val = np.array([fold["f1_macro"] for fold in val_metrics])

    # ------------------------------------------------------
    # Build output directory
    #
    # Example:
    # Results_new/num_nodes_20/logs/COV+ATM/SimpleNN
    #
    # becomes
    #
    # Results_new/NN/num_nodes_20/COV+ATM/SimpleNN
    # ------------------------------------------------------
    relative = os.path.relpath(root, old_results)

    parts = relative.split(os.sep)

    # Remove "logs" if present
    parts = [p for p in parts if p != "logs"]

    # Insert "NN" after the num_nodes_* folder
    if len(parts) >= 2:
        parts.insert(1, "NN")

    out_dir = os.path.join(new_results, *parts)
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------
    # Save in standardized format
    # ------------------------------------------------------
    np.save(os.path.join(out_dir, "cv_balanced_accuracy_train.npy"),
        bal_acc_train)

    np.save(os.path.join(out_dir, "cv_balanced_accuracy_val.npy"),
        bal_acc_val)

    np.save(os.path.join(out_dir, "cv_f1_train.npy"),
        f1_train)

    np.save(os.path.join(out_dir, "cv_f1_val.npy"),
        f1_val)

    # ------------------------------------------------------
    # Print summary
    # ------------------------------------------------------
    print(f"\nConverted: {relative}")
    print(f"  Balanced Accuracy (val): "
        f"{bal_acc_val.mean():.4f} ± {bal_acc_val.std():.4f}")
    print(f"  Balanced Accuracy (train): "
        f"{bal_acc_train.mean():.4f} ± {bal_acc_train.std():.4f}")
    print(f"  F1 (val): "
        f"{f1_val.mean():.4f} ± {f1_val.std():.4f}")
    print(f"  F1 (train): "
        f"{f1_train.mean():.4f} ± {f1_train.std():.4f}")

print("\nDone! All SimpleNN and DeepNN results have been converted.")