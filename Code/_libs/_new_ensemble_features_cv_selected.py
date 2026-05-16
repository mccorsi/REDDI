from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
from select_features import FC_DimRed
from utils import upper_triangular_flatten

class EnsembleClassifier:
    def __init__(self, model1, model2, metrics=None, feature_names=['covariance_matrices', 'atms']):
        self.model1 = model1
        self.model2 = model2
        self.metrics = metrics or {
            'accuracy': accuracy_score,
            'balanced_accuracy': balanced_accuracy_score,
            'f1_macro': lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'),
        }
        self.feature_names = feature_names  
        self.models_1 = []  # List to store models trained for each fold, for model 1
        self.models_2 = []  # List to store models trained for each fold, for model 2

        self._init_metrics()  # Initialize metrics for validation and training
    
    def _init_metrics(self):
        # Validation metrics
        self.validation_metrics_ensemble = []  # List to store metrics for each fold
        self.validation_metrics_1 = []  # List to store metrics for model 1
        self.validation_metrics_2 = []  # List to store metrics for model 2

        # For training metrics
        self.train_metrics_ensemble = [] 
        self.train_metrics_1 = [] 
        self.train_metrics_2 = [] 

    def fit(self, X_1, X_2, y, cv, num_nodes_feat_selection=None, threshold_feat_selection=None, riemanian_classifier=False):
        self.models_1 = []
        self.models_2 = []

        self._init_metrics() # Reset metrics for each fit call

        self.oof_preds = np.zeros_like(y)  # Out-of-fold predictions. Useful for the confusion matrix evaluation
        self.oof_preds_1 = np.zeros_like(y)  
        self.oof_preds_2 = np.zeros_like(y)
        self.true_labels = np.zeros_like(y)

        self.oof_probabilities_1 = np.zeros((len(y), 4))
        self.oof_probabilities_2 = np.zeros((len(y), 4))

        for fold, (train_idx, val_idx) in enumerate(cv.split(X_1, y)):
            

            X_1_train, X_1_val = X_1[train_idx], X_1[val_idx]
            X_2_train, X_2_val = X_2[train_idx], X_2[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            if len(X_1_train.shape) == 3:
                # If the input is 3D (e.g., FC matrices), we perform dimensionality reduction before flattening
                # Dimensionality reduction
                dim_red_eta_1 = FC_DimRed(eta_threshold=threshold_feat_selection, nb_nodes=num_nodes_feat_selection)
                X_1_train = dim_red_eta_1.fit_transform(X_1_train, y_train, metric='eta-squared')
                X_1_val = dim_red_eta_1.transform(X_1_val)
                print(f"Selected nodes First Matrices: {dim_red_eta_1.node_select_}")

                dim_red_eta_2 = FC_DimRed(eta_threshold=threshold_feat_selection, nb_nodes=num_nodes_feat_selection)
                X_2_train = dim_red_eta_2.fit_transform(X_2_train, y_train, metric='eta-squared')
                X_2_val = dim_red_eta_2.transform(X_2_val)
                print(f"Selected nodes Second Matrices: {dim_red_eta_2.node_select_}")

                if riemanian_classifier == False:
                    # Flatten the matrices after dimensionality reduction
                    print("Dimension of First Matrices before flattening: ", X_1_train.shape)
                    print("Dimension of Second Matrices before flattening: ", X_2_train.shape)
                    X_1_train = np.array([upper_triangular_flatten(mat) for mat in X_1_train])
                    X_2_train = np.array([upper_triangular_flatten(mat) for mat in X_2_train])
                    X_1_val = np.array([upper_triangular_flatten(mat) for mat in X_1_val])
                    X_2_val = np.array([upper_triangular_flatten(mat) for mat in X_2_val])
                    print("Dimension of First Matrices after flattening: ", X_1_train.shape)
                    print("Dimension of Second Matrices after flattening: ", X_2_train.shape)

            # Train each model on their respective feature sets
            model1 = self.model1
            model2 = self.model2

            model1.fit(X_1_train, y_train)
            model2.fit(X_2_train, y_train)

            # Store the trained models for each fold
            self.models_1.append(model1)
            self.models_2.append(model2)

            ####### TRAINING METRICS #######
            train_preds_ensemble, train_preds_1, train_preds_2, _, _ = self._predict_fold(model1, model2, X_1_train, X_2_train)
            # Ensemble
            train_metrics_ensemble = {name: fn(y_train, train_preds_ensemble) for name, fn in self.metrics.items()}
            self.train_metrics_ensemble.append(train_metrics_ensemble)
            # Model 1 
            train_metrics_1 = {name: fn(y_train, train_preds_1) for name, fn in self.metrics.items()}
            self.train_metrics_1.append(train_metrics_1)
            # Model 2 
            train_metrics_2 = {name: fn(y_train, train_preds_2) for name, fn in self.metrics.items()}
            self.train_metrics_2.append(train_metrics_2)

            ####### VALIDATION METRICS #######
            ensemble_preds, validation_preds_1, validation_preds_2, output_probabilities_1 , output_probabilities_2 = self._predict_fold(model1, model2, X_1_val, X_2_val)
            # Ensemble
            metrics = {name: fn(y_val, ensemble_preds) for name, fn in self.metrics.items()}
            self.validation_metrics_ensemble.append(metrics)
            # Model 1 metrics
            validation_metrics_1 = {name: fn(y_val, validation_preds_1) for name, fn in self.metrics.items()}
            self.validation_metrics_1.append(validation_metrics_1)
            # Model 2 metrics
            validation_metrics_2 = {name: fn(y_val, validation_preds_2) for name, fn in self.metrics.items()}
            self.validation_metrics_2.append(validation_metrics_2)

            # Store out-of-fold predictions
            self.oof_preds[val_idx] = ensemble_preds
            self.oof_preds_1[val_idx] = validation_preds_1
            self.oof_preds_2[val_idx] = validation_preds_2
            # Store true labels for the validation set
            self.true_labels[val_idx] = y_val
            # Store out-of-fold probabilities
            self.oof_probabilities_1[val_idx] = output_probabilities_1
            self.oof_probabilities_2[val_idx] = output_probabilities_2

            print(f"Fold {fold + 1} ensemble metrics: {metrics}")

    def _predict_fold(self, model1, model2, X_1, X_2):
        proba1 = model1.predict_proba(X_1)
        proba2 = model2.predict_proba(X_2)

        final_preds = []
        preds_1 = []
        output_probabilities_1 = []
        preds_2 = []
        output_probabilities_2 = []
        for p1, p2 in zip(proba1, proba2):
            pred1 = np.argmax(p1)
            pred2 = np.argmax(p2)
            preds_1.append(pred1) # Store predictions for model 1
            output_probabilities_1.append(p1) # Store probabilities for model 1
            output_probabilities_2.append(p2) # Store probabilities for model 2
            preds_2.append(pred2) # Store predictions for model 2
            if pred1 == pred2:
                final_preds.append(pred1)
            else:
                # Choose the one with higher confidence
                if self.feature_names[1] == 'atms':
                    if np.max(p1) >= np.max(p2) - 0.2: # ATMs are more prone to overfitting, so we lower the threshold of confidence
                        final_preds.append(pred1)
                    else:
                        final_preds.append(pred2)
                else:
                    if np.max(p1) >= np.max(p2):
                        final_preds.append(pred1)
                    else:
                        final_preds.append(pred2)

        return np.array(final_preds), np.array(preds_1), np.array(preds_2), np.array(output_probabilities_1), np.array(output_probabilities_2)

    def get_cv_scores(self):
        results = {}
        for metric_name in self.metrics.keys():
            scores = [metrics[metric_name] for metrics in self.validation_metrics_ensemble]
            results[metric_name] = {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "all": scores,
            }
        return results

    def predict(self, X_1, X_2):
        all_preds = []  # Will be 5(if cv=5) x N array, where N is the number of samples
        for model1, model2 in zip(self.models_1, self.models_2):
            preds, _ , _ , _, _= self._predict_fold(model1, model2, X_1, X_2)
            all_preds.append(preds)  

        # Majority vote across CV models
        all_preds = np.stack(all_preds)
        final_preds = []
        for i in range(all_preds.shape[1]):
            votes = all_preds[:, i]
            final_preds.append(np.bincount(votes).argmax())
        return np.array(final_preds)
    
    def plot_validation_metrics(self, output_path="results"):
        os.makedirs(output_path, exist_ok=True)

        # Plot all metrics on the same plot, each metric's fold scores on a vertical line
        metric_names = list(self.metrics.keys())
        plt.figure(figsize=(7, 6))
        for idx, metric_name in enumerate(metric_names):
            scores = [metrics[metric_name] for metrics in self.validation_metrics_ensemble]
            x = np.full(len(scores), idx + 1)  # All scores for this metric on the same vertical line
            plt.scatter(x, scores, label=f"{metric_name} folds")
            # Plot mean and std for each metric
            mean = np.mean(scores)
            std = np.std(scores)
            plt.errorbar(idx + 1, mean, yerr=std, fmt='D', capsize=5)
        plt.xticks(range(1, len(metric_names) + 1), metric_names)
        plt.xlabel("Metric")
        plt.ylabel("Score")
        plt.title("Cross-Validation Scores per Metric - Ensemble Model")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "cv_metrics_ensemble.png"))
        plt.show()

        plt.figure(figsize=(7, 6))
        for idx, metric_name in enumerate(metric_names):
            scores = [metrics[metric_name] for metrics in self.validation_metrics_1]
            x = np.full(len(scores), idx + 1)
            plt.scatter(x, scores, label=f"{metric_name} folds")
            mean = np.mean(scores)
            std = np.std(scores)
            plt.errorbar(idx + 1, mean, yerr=std, fmt='D', capsize=5)
        plt.xticks(range(1, len(metric_names) + 1), metric_names)
        plt.xlabel("Metric")
        plt.ylabel("Score")
        plt.title(f"Cross-Validation Scores per Metric - {self.feature_names[0]} Model")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"cv_validation_metrics_{self.feature_names[0]}.png"))
        plt.show()

        plt.figure(figsize=(7, 6))
        for idx, metric_name in enumerate(metric_names):
            scores = [metrics[metric_name] for metrics in self.validation_metrics_2]
            x = np.full(len(scores), idx + 1)
            plt.scatter(x, scores, label=f"{metric_name} folds")
            mean = np.mean(scores)
            std = np.std(scores)
            plt.errorbar(idx + 1, mean, yerr=std, fmt='D', capsize=5)
        plt.xticks(range(1, len(metric_names) + 1), metric_names)
        plt.xlabel("Metric")
        plt.ylabel("Score")
        plt.title(f"Cross-Validation Scores per Metric - {self.feature_names[1]} Model")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"cv_validation_metrics_{self.feature_names[1]}.png"))
        plt.show()

    def plot_oof_confusion_matrices(self, y, output_path="results"):
        """
        Plot Out-of-Fold Confusion Matrices for ensemble, model 1, and model 2.
        """
        # Ensemble model confusion matrix
        conf_matrix_ensemble = confusion_matrix(y, self.oof_preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(conf_matrix_ensemble, annot=True, fmt="d", cmap="Blues",
                    xticklabels=['MCI', 'MS', 'PD', 'SLA'], yticklabels=['MCI', 'MS', 'PD', 'SLA'], cbar=False)
        plt.title("Ensemble Model OOF Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "oof_confusion_matrix_ensemble.png"))
        plt.show()

        # Covariance matrices model confusion matrix
        conf_matriX_1 = confusion_matrix(y, self.oof_preds_1)
        plt.figure(figsize=(6, 5))
        sns.heatmap(conf_matriX_1, annot=True, fmt="d", cmap="Blues",
                    xticklabels=['MCI', 'MS', 'PD', 'SLA'], yticklabels=['MCI', 'MS', 'PD', 'SLA'], cbar=False)
        plt.title(f"{self.feature_names[0]} Model OOF Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"oof_confusion_matrix_{self.feature_names[0]}.png"))
        plt.show()

        # ATMs model confusion matrix
        conf_matriX_2 = confusion_matrix(y, self.oof_preds_2)
        plt.figure(figsize=(6, 5))
        sns.heatmap(conf_matriX_2, annot=True, fmt="d", cmap="Blues",
                    xticklabels=['MCI', 'MS', 'PD', 'SLA'], yticklabels=['MCI', 'MS', 'PD', 'SLA'], cbar=False)
        plt.title(f"{self.feature_names[1]} Model OOF Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"oof_confusion_matrix_{self.feature_names[1]}.png"))
        plt.show()

    def save_metrics(self, output_path="results"):
        """
        Save all metrics to a JSON file.
        """
        os.makedirs(output_path, exist_ok=True)
        metrics = {
            # Validation metrics
            "validation_metrics_ensemble": self.validation_metrics_ensemble,
            f"validation_metrics_{self.feature_names[0]}": self.validation_metrics_1,
            f"validation_metrics_{self.feature_names[1]}": self.validation_metrics_2,
            # Train metrics
            "train_metrics_ensemble": self.train_metrics_ensemble,
            f"train_metrics_{self.feature_names[0]}": self.train_metrics_1,
            f"train_metrics_{self.feature_names[1]}": self.train_metrics_2,
        }
        with open(os.path.join(output_path, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=4)

    def store_oof_probabilities(self, output_path):
        """
        Stores the out-of-fold probabilities and predicted classes for the models 1 and 2 to a file.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data = {
            f"{self.feature_names[0]}": {
                "true_classes": self.true_labels.tolist(),
                "predicted_classes": self.oof_preds_1.tolist(),
                "probabilities": self.oof_probabilities_1.tolist()
            },
            f"{self.feature_names[1]}": {
                "true_classes": self.true_labels.tolist(),
                "predicted_classes": self.oof_preds_2.tolist(),
                "probabilities": self.oof_probabilities_2.tolist()
            }
        }
        with open(os.path.join(output_path, "probabilities.json"), "w") as f:
            json.dump(data, f, indent=4)
