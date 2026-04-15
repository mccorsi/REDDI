import json
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os
import sys

###################################
# Used to produce figure 6C
###################################

json_path = 'Results/logs/Corr_Mat/TSClassifier/probabilities.json'
current_dir = os.path.dirname(os.path.abspath(__file__))
print( current_dir)
json_path = os.path.join(current_dir, "..","..",json_path)
save_dir = os.path.join(current_dir, "..","..","Images")
print(json_path)
os.makedirs(save_dir, exist_ok=True)

# Load data
with open(json_path, 'r') as f:
    data = json.load(f)

cov = data['covariance_matrices']
corr = data['correlation_matrices']

true_classes = cov['true_classes']
cov_probs = cov['probabilities']
corr_probs = corr['probabilities']

num_samples = len(true_classes)
num_classes = len(cov_probs[0])

for i in range(num_samples):
    # Binarize the true classes for ROC computation
    y_true = label_binarize(true_classes, classes=list(range(num_classes)))

    # Collect the probabilities chosen for each sample
    chosen_probs = []
    for i in range(num_samples):
        cov_max = np.max(cov_probs[i])
        corr_max = np.max(corr_probs[i])
        if cov_max >= corr_max:
            chosen_probs.append(cov_probs[i])
        else:
            chosen_probs.append(corr_probs[i])
    chosen_probs = np.array(chosen_probs)

class_labels = ['MCI', 'MS', 'PD', 'SLA']

# Plot ROC curve for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], chosen_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    # Inspect unique probabilities for each class
    unique_probs = np.unique(chosen_probs[:, i])
    print(f"Unique probabilities for class {class_labels[i]}: {unique_probs}")

plt.figure()
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {class_labels[i]} (AUC = {roc_auc[i]:.2f})')



plt.plot([0, 1], [0, 1], 'k--', lw=0.8)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Final Predictions')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(save_dir+"TSClassifier_roc_curves_final_predictions.png")
plt.show()