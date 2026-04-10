import os, sys, numpy as np, warnings
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath("Code/_libs"))

from utils import load_cov_mats, load_corr_mats, load_atms, upper_triangular_flatten
from select_features import FC_DimRed

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
# Helper for feature processing
# ===============================================================
def prepare_features(X, y, nb_nodes, label):
    if X.ndim == 3 and X.shape[1] >= 78:
        X = X[:, :78, :78]
    if nb_nodes is not None:
        dim_red = FC_DimRed(eta_threshold=0.1, nb_nodes=nb_nodes)
        X = dim_red.fit_transform(X, y, metric="eta-squared")
        print(f"{label}: selected {len(dim_red.node_select_)} nodes")

    # Flatten
    X = np.array([upper_triangular_flatten(m) for m in X])
    print(f"{label}: flattened shape {X.shape}")

    # Normalize to zero mean / unit variance
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print(f"{label}: scaled mean {X.mean():.4f}, std {X.std():.4f}")
    return X


# ===============================================================
# Load and encode datasets
# ===============================================================
print("\n===== Loading datasets =====")
X_cov, y_cov = load_cov_mats()
X_corr, y_corr = load_corr_mats()
X_atm,  y_atm  = load_atms()

# Ensure all have same subjects and labels
assert np.all(y_cov == y_corr) and np.all(y_cov == y_atm), "Label mismatch between datasets!"

# Encode labels to integers for Keras
le = LabelEncoder()
y_cov = le.fit_transform(y_cov)
y_corr = le.transform(y_corr)
y_atm  = le.transform(y_atm)
y = y_cov  # unified labels

# ===============================================================
# Feature selection
# ===============================================================
print("\n===== Applying feature selection =====")
X_cov  = prepare_features(X_cov,  y, nb_nodes=50, label="Covariance")
X_corr = prepare_features(X_corr, y, nb_nodes=20, label="Correlation")
X_atm  = prepare_features(X_atm,  y, nb_nodes=20, label="ATM")

# ===============================================================
# Build ensembles (feature concatenation)
# ===============================================================

X_cov_atm  = np.concatenate([X_cov,  X_atm],  axis=1)
X_cov_corr = np.concatenate([X_cov, X_corr], axis=1)

datasets = {
    "COV+ATM":  X_cov_atm,
    "COV+CORR": X_cov_corr
}

# ===============================================================
# Training setup
# ===============================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
num_classes = len(np.unique(y))

# ===============================================================
# Train Simple NN & Deep NN on each ensemble dataset
# ===============================================================
for name, X in datasets.items():
    print(f"\n===== Ensemble dataset: {name} =====")
    print("Input shape:", X.shape)

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
    Simple_out_dir = f"Results_new/NN/{name}/SimpleNN/"
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

    print(f"{name} Simple NN CV Balanced Acc validation: {bal_acc_val_deep.mean():.4f} ± {bal_acc_val_deep.std():.4f}")
    print(f"{name} Simple NN CV Balanced Acc training: {bal_acc_train_deep.mean():.4f} ± {bal_acc_train_deep.std():.4f}")
    print(f"{name} Simple NN CV F1 validation: {f1_val_deep.mean():.4f} ± {f1_val_deep.std():.4f}")
    print(f"{name} Simple NN CV F1 training: {f1_train_deep.mean():.4f} ± {f1_train_deep.std():.4f}")

    # Save
    Deep_out_dir = f"Results_new/NN/{name}/DeepNN/"
    os.makedirs(Deep_out_dir, exist_ok=True)

    np.save(os.path.join(Deep_out_dir, "cv_balanced_accuracy_val.npy"), bal_acc_val_deep)
    np.save(os.path.join(Deep_out_dir, "cv_balanced_accuracy_train.npy"), bal_acc_train_deep)
    np.save(os.path.join(Deep_out_dir, "cv_f1_val.npy"), f1_val_deep)
    np.save(os.path.join(Deep_out_dir, "cv_f1_train.npy"), f1_train_deep)



    print(f"\nFinished {name}: Results saved in Results_new/NN/{name}/")
