import json
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

###################################
# Used to produce figure 6B
###################################

nodes = 50
json_path = f'Results/num_nodes_{nodes}/logs/Corr_Mat/TSClassifier/probabilities.json'
current_dir = os.path.dirname(os.path.abspath(__file__))
#print( current_dir)
json_path = os.path.join(current_dir, "..","..",json_path)
save_dir = os.path.join(current_dir, "..","..","Images/")
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

final_predictions = []
for i in range(num_samples):
    cov_pred = np.argmax(cov_probs[i])
    corr_pred = np.argmax(corr_probs[i])
    cov_max = np.max(cov_probs[i])
    corr_max = np.max(corr_probs[i])
    if cov_max >= corr_max:
        final_predictions.append(cov_pred)
        probababilities = cov_probs[i]
    else:
        final_predictions.append(corr_pred)
        probabilities = corr_probs[i]

# Compute average probabilities for each true-predicted class pair
avg_probs = np.zeros((num_classes, num_classes))

for i in range(num_samples):
    true_cls = true_classes[i]
    pred_cls = final_predictions[i]
    # Use the probabilities from the chosen model
    if np.max(cov_probs[i]) >= np.max(corr_probs[i]):
        probs = cov_probs[i]
    else:
        probs = corr_probs[i]
    avg_probs[true_cls, pred_cls] += probs[pred_cls]

# Count occurrences for normalization
counts = np.zeros((num_classes, num_classes))
for i in range(num_samples):
    true_cls = true_classes[i]
    pred_cls = final_predictions[i]
    counts[true_cls, pred_cls] += 1

# Avoid division by zero
with np.errstate(divide='ignore', invalid='ignore'):
    avg_probs = np.divide(avg_probs, counts, out=np.zeros_like(avg_probs), where=counts!=0)

class_labels = ['MCI', 'MS', 'PD', 'SLA']

# Plot the matrix
plt.figure(figsize=(8, 6))
im = plt.imshow(avg_probs * 100, interpolation='nearest', cmap='Greens')
plt.title('Average Probability (%) for True vs Predicted Classes')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.colorbar(im, label='Average Probability (%)')
plt.xticks(np.arange(num_classes), class_labels)
plt.yticks(np.arange(num_classes), class_labels)

# Annotate each cell with the percentage value
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, f"{avg_probs[i, j]*100:.1f}%", ha="center", va="center", color="black")

#plt.title(f"Num nodes {nodes}", fontsize=14)
plt.savefig(save_dir + f"TSClassifier_average_probabilities_nodes_{nodes}.png", bbox_inches='tight')
print("Figure saved to:", save_dir + f"TSClassifier_average_probabilities_nodes_{nodes}.png")

plt.show()