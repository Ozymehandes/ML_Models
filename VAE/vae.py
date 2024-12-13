import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from model import *
SEED = 101 #23
if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # take a stratified subset of the training data, keeping only 5000 samples, with 500 samples per class
    train_targets = train_dataset.targets
    train_idx, _ = train_test_split(range(len(train_targets)), train_size=20000, stratify=train_targets, random_state=SEED)
    train_dataset = torch.utils.data.Subset(train_dataset, train_idx)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Get the indices of the items of each class in train dataset
    indices_per_class_train = {i: [] for i in range(10)}
    for i, (_, label) in enumerate(train_dataset):
        indices_per_class_train[label].append(i)
    
    indices_per_class_valid = {i: [] for i in range(10)}
    for i, (_, label) in enumerate(test_dataset):
        indices_per_class_valid[label].append(i)

    # Training an amortized VAE
    amortized_vae = ConvVAE (latent_dim=200)
    optimizer = optim.Adam(amortized_vae.parameters(), lr=0.001)
    num_of_epochs = 30
    criterion = nn.MSELoss()


    # Get a random image from each class in train and valid dataset
    images_indices_train = []
    images_indices_valid = []

    for i in range(10):
        train_ind = random.choice(indices_per_class_train[i])
        valid_ind = random.choice(indices_per_class_valid[i])
        images_indices_train.append(train_ind)
        images_indices_valid.append(valid_ind)

    recon_indices = [1, 5, 10, 20, 30]
    train_losses = []
    valid_losses = []
    for epoch in range(1, num_of_epochs+1):
        amortized_vae.train()
        ep_loss = []
        for i, (images, _) in enumerate(train_loader):
            optimizer.zero_grad()
            recon_images, mu, logvar = amortized_vae(images)
            KL_div = torch.mean(mu.pow(2) + torch.exp(logvar) - logvar - 1)
            loss = criterion(recon_images, images) + KL_div

            loss.backward()
            optimizer.step()
            ep_loss.append(loss.item())

        train_loss = np.mean(ep_loss)
        train_losses.append(train_loss)

        # Save the model
        if epoch in recon_indices:
            torch.save(amortized_vae.state_dict(), f"vae_epoch_{epoch}.pth")

        with torch.no_grad():
            amortized_vae.eval()
            ep_loss = []
            for i, (images, _) in enumerate(test_loader):
                recon_images, mu, logvar = amortized_vae(images)
                KL_div = torch.mean(mu.pow(2) + torch.exp(logvar) - logvar - 1)
                loss = criterion(recon_images, images) + KL_div
                ep_loss.append(loss.item())

            valid_loss = np.mean(ep_loss)
            valid_losses.append(valid_loss)

            print(f"Epoch {epoch}/{num_of_epochs}, Train Loss: {train_loss}, Validation Loss: {valid_loss}")

            if epoch in recon_indices:
                fig, axs = plt.subplots(10, 4, figsize=(20, 50))  # Create a figure with a 10x4 grid of subplot axes
                for i in range(10):
                    train_img, label = train_dataset[images_indices_train[i]]
                    valid_img, _ = test_dataset[images_indices_valid[i]]

                    train_img = train_img.unsqueeze(0)
                    valid_img = valid_img.unsqueeze(0)

                    mu_train, logvar_train = amortized_vae.encode(train_img)
                    mu_valid, logvar_valid = amortized_vae.encode(valid_img)
                    recon_train_img = amortized_vae.decode(mu_train)
                    recon_valid_img = amortized_vae.decode(mu_valid)

                    # Plot train images and reconstructions
                    axs[i, 0].imshow(train_img.squeeze().detach().numpy(), cmap='gray')
                    axs[i, 0].set_title(f"Train: Original Image - Class {label}, epoch {epoch}")
                    axs[i, 0].axis('off')

                    axs[i, 1].imshow(recon_train_img.squeeze().detach().numpy(), cmap='gray')
                    axs[i, 1].set_title(f"Train: Reconstructed Image - Class {label}, epoch {epoch}")
                    axs[i, 1].axis('off')

                    # Plot validation images and reconstructions
                    axs[i, 2].imshow(valid_img.squeeze().detach().numpy(), cmap='gray')
                    axs[i, 2].set_title(f"Validation: Original Image - Class {label}, epoch {epoch}")
                    axs[i, 2].axis('off')

                    axs[i, 3].imshow(recon_valid_img.squeeze().detach().numpy(), cmap='gray')
                    axs[i, 3].set_title(f"Validation: Reconstructed Image - Class {label}, epoch {epoch}")
                    axs[i, 3].axis('off')

                plt.tight_layout()
                plt.savefig(f'Q1_epoch_{epoch}_F.png')  # Save the combined plot as a PNG file
                plt.close()  # Close the figure
            

    # Plot the losses
    plt.plot(train_losses, label='Train Loss', color='r')
    plt.plot(valid_losses, label='Validation Loss', color='b')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and validation loss over epochs')
    plt.legend()
    plt.savefig(f'Q1-Losses.png')
    plt.close() 



    # Generate samples through latent variables
    amortized_vae = ConvVAE (latent_dim=200)
    z = torch.randn(10, 200)

    for epoch in recon_indices:
        with torch.no_grad():
            # Load the model from the specified epoch
            amortized_vae.load_state_dict(torch.load(f"vae_epoch_{epoch}.pth"))
            amortized_vae.eval()

            # Generate images by passing the latent variables through the decoder
            for i in range(z.size(0)):
                latent_var = z[i]
                generated_image = amortized_vae.decode(latent_var.unsqueeze(0))
                plt.subplot(2, 5, i+1)
                plt.imshow(generated_image.squeeze().detach().numpy(), cmap='gray')
                plt.axis('off')

            plt.suptitle(f"Generated images at epoch {epoch}")
            plt.savefig(f'Q2-epoch={epoch}_latent_recon_F.png')
            plt.close() 


    # Train a generator by Variational Inference, using Latent Optimization for optimizing the q vectors instead of a shared encoder.
    # Initialize the q vectors by sampling from a gaussian distribution of q ∼ N (0, I). This will be our prior distribution for this experiment.
    generator = VAEDecoder(latent_dim=200)
    q_mu = torch.randn(len(train_dataset), 200, requires_grad=True)
    q_sigma = torch.randn(len(train_dataset), 200)
    q_r = q_sigma.pow(2)
    # Add a small value to avoid log(0) for numerical stability
    q_r = torch.log(q_r + 1e-10)
    q_r.requires_grad_()

    class IndexedDataset(torch.utils.data.Dataset):
        def __init__(self, original_dataset):
            self.original_dataset = original_dataset

        def __getitem__(self, index):
            data, target = self.original_dataset[index]
            return index, data, target

        def __len__(self):
            return len(self.original_dataset)
    indexed_train_dataset = IndexedDataset(train_dataset)
    latent_train_loader = DataLoader(indexed_train_dataset, batch_size=64, shuffle=False)

    optimizer = optim.Adam([
    {'params': generator.parameters(), 'lr': 0.001},
    {'params': [q_mu], 'lr': 0.01},
    {'params': [q_r], 'lr': 0.01}])


    
    num_of_epochs = 30
    criterion = nn.MSELoss()
    train_losses = []
    for epoch in range(1, num_of_epochs+1):
        generator.train()
        ep_loss = []
        for i, (indices, images, _) in enumerate(latent_train_loader):
            optimizer.zero_grad()
            mu = q_mu[indices]
            logvar = q_r[indices]
            
            recon_images = generator(mu, logvar)

            KL_div = torch.mean(mu.pow(2) + torch.exp(logvar) - logvar - 1)
            loss = criterion(recon_images, images) + KL_div

            loss.backward()
            optimizer.step()
            ep_loss.append(loss.item())

        train_loss = np.mean(ep_loss)
        train_losses.append(train_loss)
        print(f"Epoch {epoch}/{num_of_epochs}, Train Loss: {train_loss}")

        generator.eval()
        if epoch in recon_indices:
            fig, axs = plt.subplots(10, 2, figsize=(10, 50))  # Create a figure with a 10x2 grid of subplots

            for i in range(10):
                idx = images_indices_train[i]
                train_img, label = train_dataset[idx]
                train_img = train_img.unsqueeze(0)

                mu = q_mu[idx].unsqueeze(0)
                logvar = q_r[idx].unsqueeze(0)
                recon_train_img = generator(mu, logvar)

                # Plot train images and reconstructions
                axs[i, 0].imshow(train_img.squeeze().detach().numpy(), cmap='gray')
                axs[i, 0].set_title(f"Train: Original Image - Class {label}, epoch {epoch}")
                axs[i, 0].axis('off')  # Remove axes

                axs[i, 1].imshow(recon_train_img.squeeze().detach().numpy(), cmap='gray')
                axs[i, 1].set_title(f"Train: Reconstructed Image - Class {label}, epoch {epoch}")
                axs[i, 1].axis('off')  # Remove axes

            plt.tight_layout()
            plt.savefig(f'Q3a_epoch_{epoch}t.png')  # Save the combined plot as a PNG file
            plt.close()  # Close the figure


    sampled_z = torch.randn(10, 200)

    # Ensure the generator is in evaluation mode
    generator.eval()

    # Generate images from the sampled latent vectors
    with torch.no_grad():
        sampled_images = generator.decode(sampled_z)

    # Plot the generated images
    plt.figure(figsize=(15, 6))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(sampled_images[i].squeeze().detach().cpu().numpy(), cmap='gray')
        plt.title(f"Sample {i + 1}")
        plt.axis('off')
    plt.savefig(f'Q3b-latent_recon.png')
    plt.close() 


    # Computing the log-probability of an image
    M = 1000
    sigma_p = 0.4

    trained_model = ConvVAE(latent_dim=200)
    trained_model.load_state_dict(torch.load("vae_epoch_30.pth"))
    trained_model.eval()

    # Sample 10 images for each digit, 5 from the training set and 5 from the test set
    q4_images_indices_train = []
    q4_images_indices_valid = []
    for i in range(10):
        train_ind = np.random.choice(indices_per_class_train[i], 5, replace=False)
        valid_ind = np.random.choice(indices_per_class_valid[i], 5, replace=False)
        q4_images_indices_train.append(train_ind)
        q4_images_indices_valid.append(valid_ind)
    
    train_images = []
    valid_images = []
    for i in range(10):
        dig_train = []
        dig_valid = []
        for j in range(5):
            train_img, _ = train_dataset[q4_images_indices_train[i][j]]
            valid_img, _ = test_dataset[q4_images_indices_valid[i][j]]
            dig_train.append(train_img)
            dig_valid.append(valid_img)
        train_images.append(dig_train)
        valid_images.append(dig_valid)
    

    def log_prob_image(vae, image, M=1000):
        normal_dist = torch.distributions.Normal(0, 1)
        mu, logvar = vae.encode(image.unsqueeze(0))
        std = torch.exp(0.5 * logvar)
        q_x = torch.distributions.Normal(mu, std)
        
        # Sample M times from the approximate posterior q(z|x)
        z = q_x.rsample((M,))
        
        # Calculate log_q(z|x)
        log_q_z_x = q_x.log_prob(z).sum(dim=-1)
        
        # Decode the samples
        recon_z = vae.decode(z)
        
        sigma_p = 0.4
        
        # Calculate log_p(x|z)
        p_x_z = torch.distributions.Normal(recon_z, sigma_p)
        log_p_x_z = p_x_z.log_prob(image).sum(dim=-1).sum(dim=-1)
        
        # Calculate log_p(z)
        log_p_z = normal_dist.log_prob(z).sum(dim=-1)
        
        # Use logsumexp for numerical stability
        log_M = torch.log(torch.tensor(M, dtype=torch.float32))
        log_prob = torch.logsumexp(log_p_x_z + log_p_z - log_q_z_x, dim=0) - log_M
        
        return log_prob
    
    # Calculate the log-probability of the images
    train_log_probs = []
    valid_log_probs = []

    for i in range(10):
        dig_log_probs_train = []
        dig_log_probs_valid = []
        for j in range(5):
            train_img = train_images[i][j]
            valid_img = valid_images[i][j]
            train_log_prob = log_prob_image(trained_model, train_img)
            valid_log_prob = log_prob_image(trained_model, valid_img)
            dig_log_probs_train.append(train_log_prob.item())
            dig_log_probs_valid.append(valid_log_prob.item())
        train_log_probs.append(dig_log_probs_train)
        valid_log_probs.append(dig_log_probs_valid)
    
    # Plot a single image from each digit, with its log-probability.
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(train_images[i][0].squeeze().detach().numpy(), cmap='gray')
        plt.title(f"{train_log_probs[i][0]:.2f}")
        plt.axis('off')
    plt.suptitle("Train Images and their log-probabilities")
    plt.savefig(f'Q4a-images_log_prob.png')
    plt.close() 

    # Present the average log-probability per digit
    avg_train_log_probs = [np.mean(dig) for dig in train_log_probs]
    avg_valid_log_probs = [np.mean(dig) for dig in valid_log_probs]

    avg_log_probs = np.array([train_probs + valid_probs for train_probs, valid_probs in zip(avg_train_log_probs, avg_valid_log_probs)])
    plt.bar(range(10), avg_log_probs, color='b')
    plt.xticks(range(10))
    plt.xlabel('Digit')
    plt.ylabel('Average log-probability')
    plt.title('Average log-probability per digit')
    plt.savefig(f'Q4b-log_prob_per_dig.png')
    plt.close() 

    # Present the average log-probability of the images from the (i) training set (ii) test set.
    avg_train_log_probs = np.array(avg_train_log_probs).mean()
    avg_valid_log_probs = np.array(avg_valid_log_probs).mean()

    print(f"Average log-probability of the training set: {avg_train_log_probs:.2f}")
    print(f"Average log-probability of the test set: {avg_valid_log_probs: .2f}")




    




                    








