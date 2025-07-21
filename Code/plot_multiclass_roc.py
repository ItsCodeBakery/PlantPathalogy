from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
import torch  # Ensure torch is imported

def plot_multiclass_roc_curve(model, data_loader, class_labels, device):
    model.eval()
    all_predicted_probabilities = []
    all_true_labels = []

    with torch.no_grad():
        for image_batch, label_batch in data_loader:
            image_batch = image_batch.to(device)
            output_logits = model(image_batch)
            probabilities = torch.softmax(output_logits, dim=1).cpu().numpy()
            all_predicted_probabilities.append(probabilities)
            all_true_labels.append(label_batch.numpy())

    all_predicted_probabilities = np.vstack(all_predicted_probabilities)
    all_true_labels = np.hstack(all_true_labels)

    # One-hot encode the true labels
    num_classes = len(class_labels)
    one_hot_true_labels = label_binarize(all_true_labels, classes=list(range(num_classes)))

    # Compute ROC curve and AUC for each class
    false_positive_rate = dict()
    true_positive_rate = dict()
    auc_score = dict()

    for class_index in range(num_classes):
        false_positive_rate[class_index], true_positive_rate[class_index], _ = roc_curve(
            one_hot_true_labels[:, class_index], all_predicted_probabilities[:, class_index]
        )
        auc_score[class_index] = auc(false_positive_rate[class_index], true_positive_rate[class_index])

    # Plot ROC curves for all classes
    plt.figure(figsize=(10, 8))
    for class_index in range(num_classes):
        readable_label = class_labels[class_index].replace('__', ' ').replace('_', ' ').title()
        plt.plot(
            false_positive_rate[class_index],
            true_positive_rate[class_index],
            lw=2,
            label=f"{readable_label} (AUC = {auc_score[class_index]:.2f})"
        )

    plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Diagonal baseline
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – Multi-Class")
    plt.legend(loc="lower right", fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("roc_curve_multiclass.png", dpi=300)
    plt.show()
