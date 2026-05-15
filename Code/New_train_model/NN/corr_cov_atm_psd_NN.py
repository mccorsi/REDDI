import os
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_validate
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

current_dir = os.path.dirname(os.path.abspath(__file__))
#print( current_dir)
libs_path = os.path.join(current_dir, "..","..","_libs")
#print(libs_path)
sys.path.append(libs_path)

save_dir_base = os.path.join(current_dir, "..","..","..","Results_new")
os.makedirs(save_dir_base, exist_ok=True)

data_path = os.path.join(current_dir, "..","..","..","data","features") 


from select_features import FC_DimRed
from utils import (
    load_cov_mats, load_corr_mats, load_atms, load_psds,
    upper_triangular_flatten
)

# ------------------------------------------------------------------
# Define model builder functions
# ------------------------------------------------------------------
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
# Load all datasets
# ------------------------------------------------------------------
datasets = {
    "Covariance": load_cov_mats(data_path),
    "Correlation": load_corr_mats(data_path),
    "ATM": load_atms(data_path),
    "PSDs": load_psds(data_path),
}

# Feature selection setup
dim_red_configs = {
    "Covariance": 50,
    "Correlation": 20,
    "ATM": 20,
    "PSDs": None  # keep all
}

# ------------------------------------------------------------------
# Training loop for each dataset and NN type
# ------------------------------------------------------------------
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


    # ---------- Simple NN ----------
    print(f"\nTraining Simple NN on {name}...")
    simple_nn = KerasClassifier(
        model=lambda: build_simple_nn(X.shape[1], num_classes),
        epochs=50,
        batch_size=16,
        verbose=0
    )

    scores_simple = cross_validate(simple_nn, X, y, cv=cv, 
                                   scoring={"bal_acc": "balanced_accuracy","f1": "f1_macro" }, 
                                   return_train_score=True)

    # Balanced accuracy
    bal_acc_train_simple = scores_simple["train_bal_acc"]
    bal_acc_val_simple   = scores_simple["test_bal_acc"]

    # F1
    f1_train_simple = scores_simple["train_f1"]
    f1_val_simple   = scores_simple["test_f1"]

    print(f"{name} Simple NN CV Balanced Acc validation: {bal_acc_val_simple.mean():.4f} ± {bal_acc_val_simple.std():.4f}")
    print(f"{name} Simple NN CV Balanced Acc training: {bal_acc_train_simple.mean():.4f} ± {bal_acc_train_simple.std():.4f}")
    print(f"{name} Simple NN CV F1 validation: {f1_val_simple.mean():.4f} ± {f1_val_simple.std():.4f}")
    print(f"{name} Simple NN CV F1 training: {f1_train_simple.mean():.4f} ± {f1_train_simple.std():.4f}")

    # Save
    Simple_out_dir = save_dir_base+f"/NN/{name}/SimpleNN/"
    os.makedirs(Simple_out_dir, exist_ok=True)

    np.save(os.path.join(Simple_out_dir, "cv_balanced_accuracy_val.npy"), bal_acc_val_simple)
    np.save(os.path.join(Simple_out_dir, "cv_balanced_accuracy_train.npy"), bal_acc_train_simple)
    np.save(os.path.join(Simple_out_dir, "cv_f1_val.npy"), f1_val_simple)
    np.save(os.path.join(Simple_out_dir, "cv_f1_train.npy"), f1_train_simple)



    # ---------- Deep NN ----------
    print(f"\nTraining Deep NN on {name}...")
    deep_nn = KerasClassifier(
        model=lambda: build_deep_nn(X.shape[1], num_classes),
        epochs=80,
        batch_size=16,
        verbose=0
    )

    
    scores_deep = cross_validate(deep_nn, X, y, cv=cv, 
                                   scoring={"bal_acc": "balanced_accuracy","f1": "f1_macro" }, 
                                   return_train_score=True)
    # Balanced accuracy
    bal_acc_train_deep = scores_deep["train_bal_acc"]
    bal_acc_val_deep   = scores_deep["test_bal_acc"]

    # F1
    f1_train_deep = scores_deep["train_f1"]
    f1_val_deep   = scores_deep["test_f1"]

    print(f"{name} Deep NN CV Balanced Acc validation: {bal_acc_val_deep.mean():.4f} ± {bal_acc_val_deep.std():.4f}")
    print(f"{name} Deep NN CV Balanced Acc training: {bal_acc_train_deep.mean():.4f} ± {bal_acc_train_deep.std():.4f}")
    print(f"{name} Deep NN CV F1 validation: {f1_val_deep.mean():.4f} ± {f1_val_deep.std():.4f}")
    print(f"{name} Deep NN CV F1 training: {f1_train_deep.mean():.4f} ± {f1_train_deep.std():.4f}")

    # Save
    Deep_out_dir = save_dir_base+f"/NN/{name}/DeepNN/"
    os.makedirs(Deep_out_dir, exist_ok=True)

    np.save(os.path.join(Deep_out_dir, "cv_balanced_accuracy_val.npy"), bal_acc_val_deep)
    np.save(os.path.join(Deep_out_dir, "cv_balanced_accuracy_train.npy"), bal_acc_train_deep)
    np.save(os.path.join(Deep_out_dir, "cv_f1_val.npy"), f1_val_deep)
    np.save(os.path.join(Deep_out_dir, "cv_f1_train.npy"), f1_train_deep)



    print(f"\nFinished {name}: Results saved in Results_new/NN/{name}/")
