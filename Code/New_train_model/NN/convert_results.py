import os
import json
import numpy as np

# ==========================================================
# Paths
# ==========================================================
# Folder containing the current (old-format) results
old_results = "Results_new"

# Folder where the standardized results will be saved
new_results = "Results_standardized"

os.makedirs(new_results, exist_ok=True)

# ==========================================================
# Search recursively for SimpleNN and DeepNN folders
# ==========================================================
for root, dirs, files in os.walk(old_results):

    # Process ONLY folders named SimpleNN or DeepNN
    model_name = os.path.basename(root)
    if model_name not in ["SimpleNN", "DeepNN"]:
        continue

    # Skip if metrics.json does not exist
    if "metrics.json" not in files:
        print(f"Skipping {root}: metrics.json not found.")
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
    bal_acc_train = np.array(
        [fold["balanced_accuracy"] for fold in train_metrics]
    )

    bal_acc_val = np.array(
        [fold["balanced_accuracy"] for fold in val_metrics]
    )

    f1_train = np.array(
        [fold["f1_macro"] for fold in train_metrics]
    )

    f1_val = np.array(
        [fold["f1_macro"] for fold in val_metrics]
    )

    # ------------------------------------------------------
    # Build output directory
    #
    # Example:
    # Results_new/num_nodes_20/logs/COV+ATM/SimpleNN
    #
    # becomes
    #
    # Results_standardized/num_nodes_20/COV+ATM/SimpleNN
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
    np.save(
        os.path.join(out_dir, "cv_balanced_accuracy_train.npy"),
        bal_acc_train,
    )

    np.save(
        os.path.join(out_dir, "cv_balanced_accuracy_val.npy"),
        bal_acc_val,
    )

    np.save(
        os.path.join(out_dir, "cv_f1_train.npy"),
        f1_train,
    )

    np.save(
        os.path.join(out_dir, "cv_f1_val.npy"),
        f1_val,
    )

    # ------------------------------------------------------
    # Print summary
    # ------------------------------------------------------
    print(f"\nConverted: {relative}")
    print(
        f"  Balanced Accuracy (val): "
        f"{bal_acc_val.mean():.4f} ± {bal_acc_val.std():.4f}"
    )
    print(
        f"  Balanced Accuracy (train): "
        f"{bal_acc_train.mean():.4f} ± {bal_acc_train.std():.4f}"
    )
    print(
        f"  F1 (val): "
        f"{f1_val.mean():.4f} ± {f1_val.std():.4f}"
    )
    print(
        f"  F1 (train): "
        f"{f1_train.mean():.4f} ± {f1_train.std():.4f}"
    )

print("\nDone! All SimpleNN and DeepNN results have been converted.")