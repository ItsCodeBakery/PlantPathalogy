import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # Avoid OpenMP hangs

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import torch  # Ensure torch is imported

def generate_tsne_plot(model, data_loader, class_labels, device, max_num_samples=10000):
    model.eval()
    all_feature_vectors = []
    all_class_labels = []

    with torch.no_grad():
        for image_batch, label_batch in data_loader:
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)

            convolutional_features = model.conv_layers(image_batch)
            flattened_features = convolutional_features.view(convolutional_features.size(0), -1)

            all_feature_vectors.append(flattened_features.cpu())
            all_class_labels.append(label_batch.cpu())

            if len(torch.cat(all_class_labels)) >= max_num_samples:
                break

    all_feature_vectors = torch.cat(all_feature_vectors)[:max_num_samples]
    all_class_labels = torch.cat(all_class_labels)[:max_num_samples]

    print(f"Running PCA → t-SNE on {all_feature_vectors.shape[0]} samples...")

    # Step 1: PCA for dimensionality reduction before t-SNE
    pca_model = PCA(n_components=50)
    reduced_features_pca = pca_model.fit_transform(all_feature_vectors)

    # Step 2: Apply t-SNE
    tsne_model = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_embedding = tsne_model.fit_transform(reduced_features_pca)

    # Prepare human-readable labels
    readable_class_labels = [class_labels[i].replace('__', ' ').replace('_', ' ').title() for i in all_class_labels]

    # Plotting
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=tsne_embedding[:, 0], y=tsne_embedding[:, 1],
        hue=readable_class_labels, palette='Spectral', s=60, alpha=0.8
    )

    plt.title("t-SNE Visualization of CNN Features", fontsize=14)
    plt.legend(loc='best', bbox_to_anchor=(1.02, 1), fontsize='small')
    plt.tight_layout()
    plt.savefig("tsnePlantVillage.png", dpi=300)
    plt.show()
