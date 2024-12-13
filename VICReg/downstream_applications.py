from train_vicreg import *
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances

def silhouette_score(X, labels):
    n_samples = X.shape[0]
    n_clusters = len(np.unique(labels))
    
    distances = euclidean_distances(X)
    silhouette_scores = np.zeros(n_samples)
    
    for i in range(n_samples):
        same_cluster_points = np.where(labels == labels[i])[0]
        if len(same_cluster_points) > 1:  # Exclude the point itself
            a_i = np.mean(distances[i, same_cluster_points[same_cluster_points != i]])
        else:
            a_i = 0
        
        b_i = np.inf
        for cluster in range(n_clusters):
            if cluster != labels[i]:
                other_cluster_points = np.where(labels == cluster)[0]
                if len(other_cluster_points) > 0:
                    cluster_distance = np.mean(distances[i, other_cluster_points])
                    b_i = min(b_i, cluster_distance)
        
        if a_i < b_i:
            silhouette_scores[i] = 1 - a_i / b_i
        elif a_i > b_i:
            silhouette_scores[i] = b_i / a_i - 1
        else:
            silhouette_scores[i] = 0
    
    return np.mean(silhouette_scores)

def plot_tsne(representations, kmeans, true_labels, model_name):
    # Combine representations and cluster centers
    combined_data = np.vstack((representations, kmeans.cluster_centers_))
    
    # Perform T-SNE on combined data
    tsne = TSNE(n_components=2, random_state=42)
    combined_2d = tsne.fit_transform(combined_data)
    
    # Separate the results
    representations_2d = combined_2d[:len(representations)]
    centers_2d = combined_2d[len(representations):]
    
    # Get cluster labels
    kmeans_labels = kmeans.predict(representations)
    
    # Plot settings
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(f'T-SNE Visualization for {model_name}', fontsize=16)
    
    # Plot colored by cluster index
    scatter1 = ax1.scatter(representations_2d[:, 0], representations_2d[:, 1], c=kmeans_labels, cmap='tab10')
    ax1.scatter(centers_2d[:, 0], centers_2d[:, 1], c='black', s=200, alpha=0.5, marker='x')
    ax1.set_title('Colored by Cluster Index')
    plt.colorbar(scatter1, ax=ax1)
    
    # Plot colored by true class index
    scatter2 = ax2.scatter(representations_2d[:, 0], representations_2d[:, 1], c=true_labels, cmap='tab10')
    ax2.scatter(centers_2d[:, 0], centers_2d[:, 1], c='black', s=200, alpha=0.5, marker='x')
    ax2.set_title('Colored by True Class Index')
    plt.colorbar(scatter2, ax=ax2)
    
    plt.tight_layout()
    plt.show()


def grayscale_to_rgb(image):
    return image.convert("RGB")

def compute_representations(model, dataloader, device):
    model.eval()
    representations = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            reps = model.encoder.encode(images)
            representations.append(reps.cpu().numpy())
    return np.concatenate(representations)

def knn_density_estimation(train_reps, test_reps, k=2):
    nn = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
    nn.fit(train_reps)
    distances, _ = nn.kneighbors(test_reps)
    return np.mean(distances[:, 1:], axis=1)  # Exclude the first neighbor (self)

def plot_roc_curve(y_true, y_score, title):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {title}')
    plt.legend(loc="lower right")
    plt.show()

def plot_anomalous_samples(images, scores, title, n=7):
    most_anomalous = np.argsort(scores)[-n:][::-1]
    plt.figure(figsize=(20, 3))
    for i, idx in enumerate(most_anomalous):
        plt.subplot(1, n, i+1)
        plt.imshow(images[idx].permute(1, 2, 0))
        plt.title(f"Score: {scores[idx]:.2f}")
        plt.axis('off')
    plt.suptitle(title)
    plt.show()


if __name__ == '__main__':
    # Anomaly detection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mnist_test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Lambda(grayscale_to_rgb),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    cifar10_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=test_transform)
    cifar10_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=mnist_test_transform)

    cifar10_train_loader = DataLoader(cifar10_train, batch_size=256, shuffle=False)
    cifar10_test_loader = DataLoader(cifar10_test, batch_size=256, shuffle=False)
    mnist_test_loader = DataLoader(mnist_test, batch_size=256, shuffle=False)

    vicreg_model = VicReg(D=128, proj_dim=512, device=device).to(device)
    vicreg_model.load_state_dict(torch.load('./vicreg_model.pth'))

    vicreg_nn_model = VicReg(D=128, proj_dim=512, device=device).to(device)
    vicreg_nn_model.load_state_dict(torch.load('./nn_vicreg_model.pth'))

    # Compute representations
    train_reps = compute_representations(vicreg_model, cifar10_train_loader, device)
    test_reps_cifar10_vicreg = compute_representations(vicreg_model, cifar10_test_loader, device)
    test_reps_mnist_vicreg = compute_representations(vicreg_model, mnist_test_loader, device)
    test_reps_cifar10_no_neighbors = compute_representations(vicreg_nn_model, cifar10_test_loader, device)
    test_reps_mnist_no_neighbors = compute_representations(vicreg_nn_model, mnist_test_loader, device)

    # Concatenate test representations
    test_reps_vicreg = np.concatenate([test_reps_cifar10_vicreg, test_reps_mnist_vicreg])
    test_reps_no_neighbors = np.concatenate([test_reps_cifar10_no_neighbors, test_reps_mnist_no_neighbors])

    # Compute anomaly scores
    scores_vicreg = knn_density_estimation(train_reps, test_reps_vicreg)
    scores_no_neighbors = knn_density_estimation(train_reps, test_reps_no_neighbors)

    # Create labels (0 for CIFAR10, 1 for MNIST)
    y_true = np.concatenate([np.zeros(len(test_reps_cifar10_vicreg)), np.ones(len(test_reps_mnist_vicreg))])

    # Plot ROC curves
    plot_roc_curve(y_true, scores_vicreg, "VICReg")
    plot_roc_curve(y_true, scores_no_neighbors, "VICReg without Generated Neighbors")

    # Plot most anomalous samples
    test_images_cifar10 = torch.cat([img for img, _ in cifar10_test_loader])
    test_images_mnist = torch.cat([img for img, _ in mnist_test_loader])
    test_images = torch.cat([test_images_cifar10, test_images_mnist], dim=0)
    plot_anomalous_samples(test_images, scores_vicreg, "Most Anomalous Samples - VICReg")
    plot_anomalous_samples(test_images, scores_no_neighbors, "Most Anomalous Samples - VICReg without Generated Neighbors")

    # Print comparison
    print(f"VICReg AUC: {auc(roc_curve(y_true, scores_vicreg)[0], roc_curve(y_true, scores_vicreg)[1]):.4f}")
    print(f"VICReg without Generated Neighbors AUC: {auc(roc_curve(y_true, scores_no_neighbors)[0], roc_curve(y_true, scores_no_neighbors)[1]):.4f}")


    # Clustering

    from sklearn.cluster import KMeans
    vicreg_model = VicReg(D=128, proj_dim=512, device=device).to(device)
    vicreg_model.load_state_dict(torch.load('/kaggle/working/vicreg_model.pth'))

    vicreg_nn_model = VicReg(D=128, proj_dim=512, device=device).to(device)
    vicreg_nn_model.load_state_dict(torch.load('/kaggle/working/nn_vicreg_model.pth'))

    def compute_clustering_representations(model, dataloader, device):
        model.eval()
        representations = []
        labels = []
        with torch.no_grad():
            for images, targets in tqdm(dataloader, desc="Computing representations"):
                images = images.to(device)
                reps = model.encoder.encode(images)
                representations.append(reps.cpu().numpy())
                labels.extend(targets.numpy())
        return np.concatenate(representations), np.array(labels)

    cifar10_train = datasets.CIFAR10(root='/kaggle/input/cifer-10', train=True, download=False, transform=test_transform)
    train_loader = DataLoader(cifar10_train, batch_size=256, shuffle=False)

    # Compute representations for both models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vicreg_reps, true_labels = compute_clustering_representations(vicreg_model, train_loader, device)
    vicreg_nn_reps, _ = compute_clustering_representations(vicreg_nn_model, train_loader, device)

    # Perform K-means clustering
    kmeans_vicreg = KMeans(n_clusters=10, random_state=42,n_init='auto')
    kmeans_vicreg_nn = KMeans(n_clusters=10, random_state=42, n_init='auto')

    vicreg_clusters = kmeans_vicreg.fit(vicreg_reps)
    vicreg_nn_clusters = kmeans_vicreg_nn.fit(vicreg_nn_reps)

    plot_tsne(vicreg_reps, kmeans_vicreg, true_labels, "VICReg")
    plot_tsne(vicreg_nn_reps, kmeans_vicreg_nn, true_labels, "VICReg_No_Neighbors")

    from sklearn.metrics import silhouette_score as sklearn_silhouette_score


    vicreg_silhouette = silhouette_score(vicreg_reps, vicreg_clusters.labels_)
    vicreg_nn_silhouette = silhouette_score(vicreg_nn_reps, vicreg_nn_clusters.labels_)

    print(f"VICReg Silhouette Score: {vicreg_silhouette:.4f}")
    print(f"VICReg NN Silhouette Score: {vicreg_nn_silhouette:.4f}")

    # Comparing with sklearn's implementation - Figured out that this exists a bit late
    sklearn_vicreg_silhouette = sklearn_silhouette_score(vicreg_reps, vicreg_clusters.labels_)
    sklearn_vicreg_nn_silhouette = sklearn_silhouette_score(vicreg_nn_reps, vicreg_nn_clusters.labels_)

    print(f"VICReg Silhouette Score (sklearn): {sklearn_vicreg_silhouette:.4f}")
    print(f"VICReg NN Silhouette Score (sklearn): {sklearn_vicreg_nn_silhouette:.4f}")