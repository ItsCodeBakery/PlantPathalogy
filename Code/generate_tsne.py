import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # Avoid OpenMP hangs

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

def generate_tsne(model, dataloader, class_names, device, max_samples=10000):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for images, lbls in dataloader:
            images = images.to(device)
            lbls = lbls.to(device)

            feats = model.conv_layers(images)
            feats = feats.view(feats.size(0), -1)  # Flatten
            features.append(feats.cpu())
            labels.append(lbls.cpu())

            if len(torch.cat(labels)) >= max_samples:
                break

    features = torch.cat(features)[:max_samples]
    labels = torch.cat(labels)[:max_samples]

    print(f"Running PCA → t-SNE on {features.shape[0]} samples...")

    # PCA before t-SNE
    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(features)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_result = tsne.fit_transform(features_pca)

    # Clean label names
    readable_labels = [class_names[i].replace('__', ' ').replace('_', ' ').title() for i in labels]

    # Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1],
                    hue=readable_labels, palette='Spectral', s=60, alpha=0.8)

    plt.title("t-SNE Visualization of CNN Features", fontsize=14)
    plt.legend(loc='best', bbox_to_anchor=(1.02, 1), fontsize='small')
    plt.tight_layout()
    plt.savefig("tsnePlantVillage.png", dpi=300)
    plt.show()
