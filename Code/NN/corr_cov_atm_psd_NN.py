import os
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_score
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

sys.path.append(os.path.abspath("Code/_libs"))


from select_features import FC_DimRed
from utils import (
    load_cov_mats, load_corr_mats, load_atms, load_psds,
    upper_triangular_flatten
)

# ------------------------------------------------------------------
# Define model builder functions -----------------------------------
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


# ------------------------------------------------------------------
# Load all datasets ------------------------------------------------

datasets = {
    "Covariance": load_cov_mats(),
    "Correlation": load_corr_mats(),
    "ATM": load_atms(),
    "PSDs": load_psds(),
}

# Feature selection setup
dim_red_configs = {
    "Covariance": 50,
    "Correlation": 20,
    "ATM": 20,
    "PSDs": None  # keep all
}

# ------------------------------------------------------------------
# Training loop for each dataset and NN type -----------------------

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, (X, y) in datasets.items():
    print(f"\n===== Dataset: {name} =====")

    # Keep consistent shape (only for 3D matrices)
    if X.ndim == 3 and X.shape[1] >= 78:
        print(f"Original shape: {X.shape}")
        X = X[:, :78, :78]


    # Apply dimensionality reduction
    nb_nodes = dim_red_configs[name]
    if nb_nodes is not None:
        dim_red = FC_DimRed(eta_threshold=0.1, nb_nodes=nb_nodes)
        X = dim_red.fit_transform(X, y, metric="eta-squared")
        print(f"Selected nodes for {name}: {dim_red.node_select_}")

    # Flatten
    #X = np.array([upper_triangular_flatten(mat) for mat in X])
    # Flatten only for 3D connectivity matrices (Cov, Corr, ATM)
    if X.ndim == 3:
        X = np.array([upper_triangular_flatten(mat) for mat in X])
        print(f"Flattened shape: {X.shape}")
    # PSDs or already-flat data → just keep as is
    elif X.ndim == 2:
        print(f"{name} already flat (no flattening applied): {X.shape}")
    else:
        raise ValueError(f"Unexpected data shape for {name}: {X.shape}")

    print(f"Flattened shape: {X.shape}")

    num_classes = len(np.unique(y))

    # ----------------------------------------------------------
    # Simple NN ------------------------------------------------

    print(f"\nTraining Simple NN on {name}...")
    simple_nn = KerasClassifier(
        model=lambda: build_simple_nn(X.shape[1], num_classes),
        epochs=50,
        batch_size=16,
        verbose=0
    )

    scores_simple = cross_val_score(simple_nn, X, y, cv=cv, scoring="accuracy")
    print(f"Simple NN CV Accuracy ({name}): {scores_simple.mean():.4f} ± {scores_simple.std():.4f}")

    # Save metrics
    simple_out_dir = f"v2/results/logs/{name}/SimpleNN/"
    os.makedirs(simple_out_dir, exist_ok=True)
    np.save(os.path.join(simple_out_dir, "cv_scores.npy"), scores_simple)

    # ----------------------------------------------------------
    # Deep NN --------------------------------------------------
    
    print(f"\nTraining Deep NN on {name}...")
    deep_nn = KerasClassifier(
        model=lambda: build_deep_nn(X.shape[1], num_classes),
        epochs=80,
        batch_size=16,
        verbose=0
    )

    scores_deep = cross_val_score(deep_nn, X, y, cv=cv, scoring="accuracy")
    print(f"Deep NN CV Accuracy ({name}): {scores_deep.mean():.4f} ± {scores_deep.std():.4f}")

    # Save metrics
    deep_out_dir = f"v2/results/logs/{name}/DeepNN/"
    os.makedirs(deep_out_dir, exist_ok=True)
    np.save(os.path.join(deep_out_dir, "cv_scores.npy"), scores_deep)

    print(f"Finished {name}: results saved under v2/results/logs/{name}/")
