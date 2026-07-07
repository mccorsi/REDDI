import os
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')
# Code is ok
from sklearn.model_selection import StratifiedKFold, cross_validate
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import balanced_accuracy_score, f1_score

current_dir = os.path.dirname(os.path.abspath(__file__))
#print( current_dir)
libs_path = os.path.join(current_dir, "..","..","_libs")
#print(libs_path)
sys.path.append(libs_path)

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

for num_nodes in [20,35,50,78]:

    save_dir_base = os.path.join(current_dir, "..","..","..","Results_new", f"num_nodes_{num_nodes}")
    os.makedirs(save_dir_base, exist_ok=True)

    # Feature selection setup
    dim_red_configs = {
        "Covariance": num_nodes,
        "Correlation": num_nodes,
        "ATM": num_nodes,
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

        nb_nodes = dim_red_configs[name]
        num_classes = len(np.unique(y))

        # Store scores manually
        bal_acc_train_list_simple, bal_acc_val_list_simple = [], []
        f1_train_list_simple, f1_val_list_simple = [], []

        bal_acc_train_list_deep, bal_acc_val_list_deep = [], []
        f1_train_list_deep, f1_val_list_deep = [], []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # -----------------------------
            # Feature selection (ONLY TRAIN)
            # -----------------------------
            if nb_nodes is not None:
                dim_red = FC_DimRed(eta_threshold=0.1, nb_nodes=nb_nodes)

                X_train = dim_red.fit_transform(X_train, y_train, metric="eta-squared")
                X_val   = dim_red.transform(X_val)

                print(f"Selected nodes (fold {fold+1}): {dim_red.node_select_}")

            # -----------------------------
            # Flatten (AFTER selection)
            # -----------------------------
            if X_train.ndim == 3:
                X_train = np.array([upper_triangular_flatten(mat) for mat in X_train])
                X_val   = np.array([upper_triangular_flatten(mat) for mat in X_val])

            elif X_train.ndim != 2:
                raise ValueError(f"Unexpected shape: {X_train.shape}")

            # -----------------------------
            # Train Simple model
            # -----------------------------
            simple_nn = build_simple_nn(X_train.shape[1], num_classes)

            simple_nn.fit(
                X_train, y_train,
                epochs=50,
                batch_size=16,
                verbose=0
            )

            y_train_pred_simple = np.argmax(simple_nn.predict(X_train, verbose=0), axis=1)
            y_val_pred_simple   = np.argmax(simple_nn.predict(X_val, verbose=0), axis=1)

            bal_acc_train_list_simple.append(balanced_accuracy_score(y_train, y_train_pred_simple))
            bal_acc_val_list_simple.append(balanced_accuracy_score(y_val, y_val_pred_simple))

            f1_train_list_simple.append(f1_score(y_train, y_train_pred_simple, average="macro"))
            f1_val_list_simple.append(f1_score(y_val, y_val_pred_simple, average="macro"))

            # -----------------------------
            # Train Deep model
            # -----------------------------

            deep_nn = build_deep_nn(X_train.shape[1], num_classes)

            deep_nn.fit(
                X_train, y_train,
                epochs=50,
                batch_size=16,
                verbose=0
            )

            y_train_pred_deep = np.argmax(deep_nn.predict(X_train, verbose=0), axis=1)
            y_val_pred_deep   = np.argmax(deep_nn.predict(X_val, verbose=0), axis=1)

            bal_acc_train_list_deep.append(balanced_accuracy_score(y_train, y_train_pred_deep))
            bal_acc_val_list_deep.append(balanced_accuracy_score(y_val, y_val_pred_deep))

            f1_train_list_deep.append(f1_score(y_train, y_train_pred_deep, average="macro"))
            f1_val_list_deep.append(f1_score(y_val, y_val_pred_deep, average="macro"))

        # Convert to arrays
        bal_acc_train_simple = np.array(bal_acc_train_list_simple)
        bal_acc_val_simple   = np.array(bal_acc_val_list_simple)
        f1_train_simple      = np.array(f1_train_list_simple)
        f1_val_simple        = np.array(f1_val_list_simple)

        bal_acc_train_deep = np.array(bal_acc_train_list_deep)
        bal_acc_val_deep   = np.array(bal_acc_val_list_deep)
        f1_train_deep      = np.array(f1_train_list_deep)
        f1_val_deep        = np.array(f1_val_list_deep)

        # -----------------------------
        # Print results
        # -----------------------------
        print(f"\n\nResults for num_nodes_{num_nodes}")
        print(f"\n{name} Simple NN CV Balanced Acc validation: {bal_acc_val_simple.mean():.4f} ± {bal_acc_val_simple.std():.4f}")
        print(f"{name} Simple NN CV Balanced Acc training: {bal_acc_train_simple.mean():.4f} ± {bal_acc_train_simple.std():.4f}")
        print(f"{name} Simple NN CV F1 validation: {f1_val_simple.mean():.4f} ± {f1_val_simple.std():.4f}")
        print(f"{name} Simple NN CV F1 training: {f1_train_simple.mean():.4f} ± {f1_train_simple.std():.4f}")

        print(f"\n{name} Deep NN CV Balanced Acc validation: {bal_acc_val_deep.mean():.4f} ± {bal_acc_val_deep.std():.4f}")
        print(f"{name} Deep NN CV Balanced Acc training: {bal_acc_train_deep.mean():.4f} ± {bal_acc_train_deep.std():.4f}")
        print(f"{name} Deep NN CV F1 validation: {f1_val_deep.mean():.4f} ± {f1_val_deep.std():.4f}")
        print(f"{name} Deep NN CV F1 training: {f1_train_deep.mean():.4f} ± {f1_train_deep.std():.4f}")



        # Save
        Simple_out_dir = save_dir_base+f"/NN/{name}/SimpleNN/"
        os.makedirs(Simple_out_dir, exist_ok=True)

        np.save(os.path.join(Simple_out_dir, "cv_balanced_accuracy_val.npy"), bal_acc_val_simple)
        np.save(os.path.join(Simple_out_dir, "cv_balanced_accuracy_train.npy"), bal_acc_train_simple)
        np.save(os.path.join(Simple_out_dir, "cv_f1_val.npy"), f1_val_simple)
        np.save(os.path.join(Simple_out_dir, "cv_f1_train.npy"), f1_train_simple)


        # Save
        Deep_out_dir = save_dir_base+f"/NN/{name}/DeepNN/"
        os.makedirs(Deep_out_dir, exist_ok=True)

        np.save(os.path.join(Deep_out_dir, "cv_balanced_accuracy_val.npy"), bal_acc_val_deep)
        np.save(os.path.join(Deep_out_dir, "cv_balanced_accuracy_train.npy"), bal_acc_train_deep)
        np.save(os.path.join(Deep_out_dir, "cv_f1_val.npy"), f1_val_deep)
        np.save(os.path.join(Deep_out_dir, "cv_f1_train.npy"), f1_train_deep)



        print(f"\nFinished {name}: Results saved in {save_dir_base+f'/NN/{name}/'}\n\n\n")
