from models import *
from augmentations import *
from tqdm.notebook import tqdm # Use tqdm instead of tqdm.notebook for console
import numpy as np
import torchvision
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms


def calculate_var_and_cov_loss(z, epsilon, gamma):
    N, D = z.shape

    # Calculate variance loss
    var = torch.var(z, dim=0)
    std = torch.sqrt(var + epsilon)
    var_loss = torch.mean(torch.max(torch.zeros_like(std), gamma - std))

    # Calculate covariance loss
    z_mean = torch.mean(z, dim=0)
    z_centered = z - z_mean.unsqueeze(0)
    cov = (z_centered.T @ z_centered) / (N - 1)
    off_diag = cov * (1 - torch.eye(D, device=cov.device))
    cov_loss = (off_diag ** 2).sum() / D

    return var_loss, cov_loss


def train_vicreg(train_set, test_set, batch_size=256, epochs=100, lr=(3*1e-4), device='cuda', mu=25, title=''):
    gamma = 1
    epsilon = 1e-4
    lambd = 25
    mu = mu
    v = 1
    projection_dim = 512
    encoder_dim = 128

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2) if test_set else None

    model = VicReg(encoder_dim, projection_dim, device).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(),lr = lr ,betas = (0.9,0.999),weight_decay = 1e-6)
    mse_loss = nn.MSELoss()


    train_total_loss = []
    train_inv_loss = []
    train_var_loss = []
    train_cov_loss = []

    test_total_loss = []
    test_inv_loss = []
    test_var_loss = []
    test_cov_loss = []

    for epoch in tqdm(range(1, epochs+1), desc="Training Progress"): 
        model.train()
        ep_train_total_loss = []
        ep_train_inv_loss = []
        ep_train_var_loss = []
        ep_train_cov_loss = []

        for i, (img1, img2, _) in tqdm(enumerate(train_loader), desc="Train Epoch Progress", total=len(train_loader), leave=False, position=0):
            img1, img2 = img1.to(device), img2.to(device)
            optimizer.zero_grad()
            z1, z2 = model(img1, img2)

            # Calculate invariant loss
            inv_loss = lambd * mse_loss(z1, z2)

            # Calculate variance and covariance loss
            var_loss1, cov_loss1 = calculate_var_and_cov_loss(z1, epsilon, gamma)
            var_loss2, cov_loss2 = calculate_var_and_cov_loss(z2, epsilon, gamma)

            var_loss = mu * (var_loss1 + var_loss2)
            cov_loss = v * (cov_loss1 + cov_loss2)

            # Calculate total loss
            loss = inv_loss + var_loss + cov_loss

            ep_train_total_loss.append(loss.item())
            ep_train_inv_loss.append(inv_loss.item())
            ep_train_var_loss.append(var_loss.item())
            ep_train_cov_loss.append(cov_loss.item())
            loss.backward()
            optimizer.step()

        train_mean_total_loss = np.mean(ep_train_total_loss)
        train_mean_inv_loss = np.mean(ep_train_inv_loss)
        train_mean_var_loss = np.mean(ep_train_var_loss)
        train_mean_cov_loss = np.mean(ep_train_cov_loss)

        train_total_loss.append(train_mean_total_loss)
        train_inv_loss.append(train_mean_inv_loss)
        train_var_loss.append(train_mean_var_loss)
        train_cov_loss.append(train_mean_cov_loss)

        tqdm.write(f'Epoch {epoch}/{epochs} Loss: {train_mean_total_loss},' 
            f'Invariant Loss: {train_mean_inv_loss}, '
            f'Variance Loss: {train_mean_var_loss}, '
            f'Covariance Loss: {train_mean_cov_loss}')
        
        if not test_set:
            continue
        ep_test_total_loss = []
        ep_test_inv_loss = []
        ep_test_var_loss = []
        ep_test_cov_loss = []

        model.eval()
        with torch.no_grad():
            for i, (img1, img2, _) in tqdm(enumerate(test_loader), desc="Test Epoch Progress", total=len(test_loader), leave=False, position=0):
                img1, img2 = img1.to(device), img2.to(device)
                z1, z2 = model(img1, img2)

                inv_loss = lambd * mse_loss(z1, z2)

                var_loss1, cov_loss1 = calculate_var_and_cov_loss(z1, epsilon, gamma)
                var_loss2, cov_loss2 = calculate_var_and_cov_loss(z2, epsilon, gamma)

                var_loss = mu * (var_loss1 + var_loss2)
                cov_loss = v * (cov_loss1 + cov_loss2)

                loss = inv_loss + var_loss + cov_loss

                ep_test_total_loss.append(loss.item())
                ep_test_inv_loss.append(inv_loss.item())
                ep_test_var_loss.append(var_loss.item())
                ep_test_cov_loss.append(cov_loss.item())

        test_mean_total_loss = np.mean(ep_test_total_loss)
        test_mean_inv_loss = np.mean(ep_test_inv_loss)
        test_mean_var_loss = np.mean(ep_test_var_loss)
        test_mean_cov_loss = np.mean(ep_test_cov_loss)

        test_total_loss.append(test_mean_total_loss)
        test_inv_loss.append(test_mean_inv_loss)
        test_var_loss.append(test_mean_var_loss)
        test_cov_loss.append(test_mean_cov_loss)

        tqdm.write(f'Test Loss: {test_mean_total_loss}, '
            f'Invariant Loss: {test_mean_inv_loss}, '
            f'Variance Loss: {test_mean_var_loss}, '
            f'Covariance Loss: {test_mean_cov_loss}')
        
    
    train_losses = {
        'total_loss': train_total_loss,
        'inv_loss': train_inv_loss,
        'var_loss': train_var_loss,
        'cov_loss': train_cov_loss
    }
    
    test_losses = {
        'total_loss': test_total_loss,
        'inv_loss': test_inv_loss,
        'var_loss': test_var_loss,
        'cov_loss': test_cov_loss
    }

    torch.save(model.state_dict(), title + 'vicreg_model.pth')
    return model, train_losses, test_losses

def plot_representations(data, labels, title):
    cifar10_classes = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab10')
    cbar = plt.colorbar(scatter, ticks=range(10))
    cbar.set_ticklabels(cifar10_classes)
    cbar.set_label('Classes')
    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()

def qualitative_retrieval_evaluation(models, dataset, num_samples=10, k_neighbors=5, k_distant=5):
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    all_representations = {name: [] for name in models.keys()}
    all_images = []
    all_labels = []
    
    for images, labels in dataloader:
        images = images.to(device)
        all_images.append(images.cpu())
        all_labels.extend(labels.tolist())
        
        for name, model in models.items():
            model.eval()
            with torch.no_grad():
                representations = model.encoder.encode(images)
                all_representations[name].append(representations.cpu())
    
    all_images = torch.cat(all_images, dim=0)
    for name in models.keys():
        all_representations[name] = torch.cat(all_representations[name], dim=0).numpy()
    
    classes = list(set(all_labels))
    selected_indices = [random.choice([i for i, label in enumerate(all_labels) if label == c]) for c in classes]
    
    for name, representations in all_representations.items():
        nn = NearestNeighbors(n_neighbors=len(representations), metric='euclidean')
        nn.fit(representations)
        
        plt.figure(figsize=(20, 4*num_samples))
        plt.suptitle(f"Retrieval Results for {name}", fontsize=16)
        
        for i, idx in enumerate(selected_indices):
            distances, indices = nn.kneighbors(representations[idx].reshape(1, -1))
            nearest_indices = indices[0][1:k_neighbors+1]  # exclude the image itself
            farthest_indices = indices[0][-k_distant:][::-1]
            
            # Plot original image
            plt.subplot(num_samples, k_neighbors + k_distant + 1, i*(k_neighbors + k_distant + 1) + 1)
            plt.imshow(all_images[idx].permute(1, 2, 0))
            plt.title(f"Image (Class {all_labels[idx]})")
            plt.axis('off')
            
            # Plot nearest neighbors
            for j, n_idx in enumerate(nearest_indices):
                plt.subplot(num_samples, k_neighbors + k_distant + 1, i*(k_neighbors + k_distant + 1) + j + 2)
                plt.imshow(all_images[n_idx].permute(1, 2, 0))
                plt.title(f"Neighbor {j+1}\n(Class {all_labels[n_idx]})")
                plt.axis('off')
            
            # Plot most distant images
            for j, f_idx in enumerate(farthest_indices):
                plt.subplot(num_samples, k_neighbors + k_distant + 1, i*(k_neighbors + k_distant + 1) + k_neighbors + j + 2)
                plt.imshow(all_images[f_idx].permute(1, 2, 0))
                plt.title(f"Distant {j+1}\n(Class {all_labels[f_idx]})")
                plt.axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    # Train original VICReg model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_set = AugmentationsDataset(root='/kaggle/input/cifer-10', train=True, transform=train_transform, download=False)
    test_set = AugmentationsDataset(root='/kaggle/input/cifer-10', train=False, transform=train_transform, download=False)
    vicreg_model, vicreg_train_losses, vicreg_test_losses = train_vicreg(train_set, test_set, batch_size=256, epochs=50, lr=(3e-4), device=device)

    # Plot losses
    for key in vicreg_train_losses.keys():
        plt.plot(vicreg_train_losses[key], label=('Train ' + key))
        plt.plot(vicreg_test_losses[key], label=('Test ' + key))
        plt.legend()
        plt.title(key)
        plt.show()
    
    # Loading trained model
    trained_model = VicReg(D=128, proj_dim=512, device=device).to(device)
    trained_model.load_state_dict(torch.load('/kaggle/working/vicreg_model.pth'))

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    repr_test_set = datasets.CIFAR10(root='/kaggle/input/cifer-10', train=False, download=False, transform=test_transform)
    repr_test_loader = DataLoader(repr_test_set, batch_size=256, shuffle=False, num_workers=2)

    representations = []
    labels = []

    with torch.no_grad():
        for images, targets in tqdm(repr_test_loader, desc="Computing representations"):
            images = images.to(device)
            batch_representations = trained_model.encoder.encode(images)
            representations.append(batch_representations.cpu().numpy())
            labels.append(targets.numpy())

    representations = np.concatenate(representations)
    labels = np.concatenate(labels)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(representations)

    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(representations)

    # Plot representations
    plot_representations(pca_result, labels, 'PCA of VICReg Representations')
    plot_representations(tsne_result, labels, 't-SNE of VICReg Representations')

    # Linear probing
    probing_train_set = datasets.CIFAR10(root='/kaggle/input/cifer-10', train=True, download=False, transform=test_transform)
    probing_test_set = datasets.CIFAR10(root='/kaggle/input/cifer-10', train=False, download=False, transform=test_transform)
    probing_train_loader = DataLoader(probing_train_set, batch_size=256, shuffle=True, num_workers=2)
    probing_test_loader = DataLoader(probing_test_set, batch_size=256, shuffle=False, num_workers=2)

    linear_probing_model = LinearProbe(trained_model.encoder, D=128, num_classes=10).to(device)
    # Freeze the encoder
    for param in linear_probing_model.encoder.parameters():
        param.requires_grad = False
    optimizer = torch.optim.Adam(linear_probing_model.fc.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    linear_probing_train_losses = []
    linear_probing_test_losses = []

    for epoch in tqdm(range(1, 31), desc="Linear Probing Training Progress"):
        linear_probing_model.train()
        ep_train_loss = []
        for i, (img, target) in tqdm(enumerate(probing_train_loader), desc="Linear Probing Train Epoch Progress", total=len(probing_train_loader), leave=False ,position=0):
            img, target = img.to(device), target.to(device)
            optimizer.zero_grad()
            output = linear_probing_model(img)
            loss = criterion(output, target)
            ep_train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        train_loss = np.mean(ep_train_loss)
        linear_probing_train_losses.append(train_loss)
        tqdm.write(f'Epoch {epoch}/30 Train Loss: {train_loss}')

        ep_test_loss = []
        ep_test_acc = 0
        linear_probing_model.eval()
        with torch.no_grad():
            for i, (img, target) in tqdm(enumerate(probing_test_loader), desc="Linear Probing Test Epoch Progress", total=len(probing_test_loader), leave=False):
                img, target = img.to(device), target.to(device)
                output = linear_probing_model(img)
                loss = criterion(output, target)
                preds = output.argmax(dim=1)
                correct_preds = (preds == target).sum().item()
                ep_test_acc += correct_preds
                ep_test_loss.append(loss.item())

        test_loss = np.mean(ep_test_loss)
        linear_probing_test_losses.append(test_loss)
        tqdm.write(f'Epoch {epoch}/30 Test Loss: {test_loss}')
        tqdm.write(f'Epoch {epoch}/30 Test Accuracy: {ep_test_acc/len(probing_test_set)}')

    linear_probing_model.eval()
    torch.save(linear_probing_model.state_dict(), 'linear_probe_model.pth')

    # Variance ablated model
    train_set = AugmentationsDataset(root='/kaggle/input/cifer-10', train=True, transform=train_transform, download=False)
    test_set = AugmentationsDataset(root='/kaggle/input/cifer-10', train=False, transform=train_transform, download=False)
    var_ablated_model, var_ablated_train_losses, var_ablated_test_losses = train_vicreg(train_set, test_set, batch_size=256, epochs=50, lr=(3*1e-4), device='cuda', mu=0, title='var_ablated_')
    trained_var_ablated_model = VicReg(D=128, proj_dim=512, device=device).to(device)
    trained_var_ablated_model.load_state_dict(torch.load('/kaggle/working/var_ablated_vicreg_model.pth'))
    var_ablated_representations = []
    var_ablated_labels = []

    with torch.no_grad():
        for images, targets in tqdm(repr_test_loader, desc="Computing representations"):
            images = images.to(device)
            batch_representations = trained_var_ablated_model.encoder.encode(images)
            var_ablated_representations.append(batch_representations.cpu().numpy())
            var_ablated_labels.append(targets.numpy())

    var_ablated_representations = np.concatenate(var_ablated_representations)
    var_ablated_labels = np.concatenate(var_ablated_labels)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(var_ablated_representations)

    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(var_ablated_representations)

    # Var ablated representations
    plot_representations(pca_result, var_ablated_labels, 'PCA of VICReg Representations Var Ablated')
    plot_representations(tsne_result, var_ablated_labels, 't-SNE of VICReg Representations Var Ablated')

    # Var ablated linear probing
    var_ablated_linear_probing_model = LinearProbe(trained_var_ablated_model.encoder, D=128, num_classes=10).to(device)
    for param in var_ablated_linear_probing_model.encoder.parameters():
        param.requires_grad = False
    optimizer = torch.optim.Adam(var_ablated_linear_probing_model.fc.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    linear_probing_train_losses = []
    linear_probing_test_losses = []

    for epoch in tqdm(range(1, 31), desc="Linear Probing Training Progress"):
        var_ablated_linear_probing_model.train()
        ep_train_loss = []
        for i, (img, target) in tqdm(enumerate(probing_train_loader), desc="Linear Probing Train Epoch Progress", total=len(probing_train_loader), leave=False ,position=0):
            img, target = img.to(device), target.to(device)
            optimizer.zero_grad()
            output = var_ablated_linear_probing_model(img)
            loss = criterion(output, target)
            ep_train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        train_loss = np.mean(ep_train_loss)
        linear_probing_train_losses.append(train_loss)
        tqdm.write(f'Epoch {epoch}/30 Train Loss: {train_loss}')

        ep_test_loss = []
        ep_test_acc = 0
        var_ablated_linear_probing_model.eval()
        with torch.no_grad():
            for i, (img, target) in tqdm(enumerate(probing_test_loader), desc="Linear Probing Test Epoch Progress", total=len(probing_test_loader), leave=False):
                img, target = img.to(device), target.to(device)
                output = var_ablated_linear_probing_model(img)
                loss = criterion(output, target)
                preds = output.argmax(dim=1)
                correct_preds = (preds == target).sum().item()
                ep_test_acc += correct_preds
                ep_test_loss.append(loss.item())

        test_loss = np.mean(ep_test_loss)
        linear_probing_test_losses.append(test_loss)
        tqdm.write(f'Epoch {epoch}/30 Test Loss: {test_loss}')
        tqdm.write(f'Epoch {epoch}/30 Test Accuracy: {ep_test_acc/len(probing_test_set)}')

    var_ablated_linear_probing_model.eval()
    torch.save(var_ablated_linear_probing_model.state_dict(), 'var_ablated_linear_probe_model.pth')

    # Nearest neighbours
    knn_train_set = datasets.CIFAR10(root='/kaggle/input/cifer-10', train=True, download=False, transform=test_transform)
    knn_test_set = datasets.CIFAR10(root='/kaggle/input/cifer-10', train=False, download=False, transform=test_transform)
    knn_train_loader = DataLoader(knn_train_set, batch_size=256, shuffle=False, num_workers=2)
    knn_test_loader = DataLoader(knn_test_set, batch_size=256, shuffle=False, num_workers=2)
    trained_vicreg_model = VicReg(D=128, proj_dim=512, device=device).to(device)
    trained_vicreg_model.load_state_dict(torch.load('/kaggle/working/vicreg_model.pth'))
    knn_representations = []
    knn_labels = []
    trained_vicreg_model.eval()
    with torch.no_grad():
        for images, targets in tqdm(knn_train_loader, desc="Computing representations"):
            images = images.to(device)
            batch_representations = trained_vicreg_model.encoder.encode(images)
            knn_representations.append(batch_representations.cpu().numpy())
            knn_labels.append(targets.numpy())
            
    knn_representations = np.concatenate(knn_representations)
    knn_labels = np.concatenate(knn_labels)

    nn_train_set = NeighborDataset(knn_train_set, knn_representations)
    nn_model, nn_train_losses, nn_test_losses = train_vicreg(nn_train_set, None, batch_size=256, epochs=50, lr=(3*1e-4), device=device, mu=25, title='nn_')

    # Saving run time by loading the model
    nn_model = VicReg(D=128, proj_dim=512, device=device).to(device)
    nn_model.load_state_dict(torch.load('/kaggle/working/nn_vicreg_model.pth'))
    nn_linear_probing_model = LinearProbe(nn_model.encoder, D=128, num_classes=10).to(device)
    for param in nn_linear_probing_model.encoder.parameters():
        param.requires_grad = False
    optimizer = torch.optim.Adam(nn_linear_probing_model.fc.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    linear_probing_train_losses = []
    linear_probing_test_losses = []

    for epoch in tqdm(range(1, 31), desc="Linear Probing Training Progress"):
        nn_linear_probing_model.train()
        ep_train_loss = []
        for i, (img, target) in tqdm(enumerate(probing_train_loader), desc="Linear Probing Train Epoch Progress", total=len(probing_train_loader), leave=False ,position=0):
            img, target = img.to(device), target.to(device)
            optimizer.zero_grad()
            output = nn_linear_probing_model(img)
            loss = criterion(output, target)
            ep_train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        train_loss = np.mean(ep_train_loss)
        linear_probing_train_losses.append(train_loss)
        tqdm.write(f'Epoch {epoch}/30 Train Loss: {train_loss}')

        ep_test_loss = []
        ep_test_acc = 0
        nn_linear_probing_model.eval()
        with torch.no_grad():
            for i, (img, target) in tqdm(enumerate(probing_test_loader), desc="Linear Probing Test Epoch Progress", total=len(probing_test_loader), leave=False):
                img, target = img.to(device), target.to(device)
                output = nn_linear_probing_model(img)
                loss = criterion(output, target)
                preds = output.argmax(dim=1)
                correct_preds = (preds == target).sum().item()
                ep_test_acc += correct_preds
                ep_test_loss.append(loss.item())

        test_loss = np.mean(ep_test_loss)
        linear_probing_test_losses.append(test_loss)
        tqdm.write(f'Epoch {epoch}/30 Test Loss: {test_loss}')
        tqdm.write(f'Epoch {epoch}/30 Test Accuracy: {ep_test_acc/len(probing_test_set)}')

    nn_linear_probing_model.eval()
    torch.save(nn_linear_probing_model.state_dict(), 'nn_linear_probe_model.pth')

    # Qualitative retrieval evaluation
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    dataset = datasets.CIFAR10(root='/kaggle/input/cifer-10', train=True, download=False, transform=test_transform)

    trained_model = VicReg(D=128, proj_dim=512, device=device).to(device)
    trained_model.load_state_dict(torch.load('/kaggle/working/vicreg_model.pth'))

    nn_model = VicReg(D=128, proj_dim=512, device=device).to(device)
    nn_model.load_state_dict(torch.load('/kaggle/working/nn_vicreg_model.pth'))
    models = {
        'Regular VICReg': trained_model,
        'KNN VICReg': nn_model
    }

    qualitative_retrieval_evaluation(models, dataset)

