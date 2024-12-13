import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
from model import ConditionalFlow, UnconditionalFlow
from create_data import *

DATA_POINTS = 250000
VALIDATION_SPLIT = 0.05
SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sample_conditonal(model, n_classes, samples_per_class, delta_t=1e-3, int_to_label=None):
    model.eval()
    normal_dist = torch.distributions.MultivariateNormal(torch.zeros(model.input_dim).to(DEVICE),
                                                        torch.eye(model.input_dim).to(DEVICE))

    plt.figure(figsize=(16, 9))

    for c in range(n_classes):
        c_tensor = torch.tensor([c]*samples_per_class, dtype=torch.long).to(DEVICE)
        y = normal_dist.sample((samples_per_class,)).to(DEVICE)
        t = 0
        while t <= 1:
            t = round(t, 3)
            t_tensor = torch.tensor([t]*samples_per_class, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
            next = model(y, t_tensor, c_tensor)*delta_t
            y = y + next
            t += delta_t

        detached = y.detach().cpu().numpy()
        plt.scatter(detached[:, 0], detached[:, 1], color=int_to_label[c])

    plt.title('Samples with conditional model')
    plt.savefig(f'olympic_rings_conditional.png')
    plt.close()

    return y

def sample_trajectory_conditional(model, n_classes, samples_per_class, delta_t=1e-3, int_to_label=None):
    model.eval()
    normal_dist = torch.distributions.MultivariateNormal(torch.zeros(model.input_dim).to(DEVICE),
                                                        torch.eye(model.input_dim).to(DEVICE))

    plt.figure(figsize=(16, 9))

    for c in range(n_classes):
        c_tensor = torch.tensor([c]*samples_per_class, dtype=torch.long).to(DEVICE)
        z = normal_dist.sample((samples_per_class,)).to(DEVICE)

        x = [[z[i, 0].detach().cpu().numpy()] for i in range(samples_per_class)]
        y = [[z[i, 1].detach().cpu().numpy()] for i in range(samples_per_class)]
        t_values = [0]
        t = 0
        while t <= 1:
            t = round(t, 3)
            t_tensor = torch.tensor([t]*samples_per_class, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
            next = model(z, t_tensor, c_tensor)*delta_t
            z = z + next
            for i in range(samples_per_class):
                x[i].append(z[i, 0].detach().cpu().numpy())
                y[i].append(z[i, 1].detach().cpu().numpy())
            t += delta_t
            t_values.append(t)
        
        for i in range(samples_per_class):
            plt.plot(x[i], y[i], color=int_to_label[c])
            # Mark the starting point with a red 'x'
            start = plt.scatter(x[i][0], y[i][0], color='red', marker='x')
            # Mark the ending point with a green 'o'
            end = plt.scatter(x[i][-1], y[i][-1], color='green', marker='o')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Trajectories of samples with conditional model')
    plt.legend([start, end], ['Start', 'End'])
    plt.savefig('olympic_rings_conditional_trajectories.png')
    plt.close()


def train_conditional_model(model, train_loader, val_loader, optimizer, scheduler, n_epochs):
    criterion = torch.nn.MSELoss()
    normal_dist = torch.distributions.MultivariateNormal(torch.zeros(model.input_dim).to(DEVICE),
                                                        torch.eye(model.input_dim).to(DEVICE))
    uniform_dist = torch.distributions.Uniform(0, 1)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = []

        for batch, labels in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            eps = normal_dist.sample((batch.shape[0],))
            t = uniform_dist.sample((batch.shape[0],)).unsqueeze(-1)
            y = t*batch + (1-t)*eps
            y_pred = model(y, t, labels.to(DEVICE))
            v_t = batch - eps
            loss = criterion(y_pred, v_t)
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())

        scheduler.step()
        
        print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {np.mean(epoch_loss):.4f}')

    return model

if __name__ == '__main__':
    # Set random seed for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Prepare data
    data, labels, int_to_label = create_olympic_rings(DATA_POINTS, verbose=False)
    data_shape = data.shape
    data = torch.tensor(data, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    train_set, val_set, train_labels, val_labels = train_test_split(data, labels, test_size=VALIDATION_SPLIT, random_state=SEED)

    # Create data loaders
    train_loader = DataLoader(TensorDataset(train_set, train_labels), batch_size=128, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_set, val_labels), batch_size=128, shuffle=False)

    # Define normal distribution
    normal_dist = torch.distributions.MultivariateNormal(torch.zeros(data_shape[1]).to(DEVICE), 
                                                         torch.eye(data_shape[1]).to(DEVICE))

    # Initialize the normalizing flow model and learning parameters
    n_classes = len(torch.unique(labels))
    model = ConditionalFlow(data_shape[1], n_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_epochs = 20
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Train the model
    model = train_conditional_model(model, train_loader, val_loader, optimizer, scheduler, n_epochs)

    # Save the model
    torch.save(model.state_dict(), 'conditional_model.pth')

    # Load the model
    conditional_model = ConditionalFlow(data_shape[1], n_classes).to(DEVICE)
    conditional_model.load_state_dict(torch.load('conditional_model.pth'))

    # Plot the data
    create_olympic_rings(5000, ring_thickness=0.25, verbose=True)

    # Sample a point for each class with their trajectory
    sample_trajectory_conditional(conditional_model, n_classes, 1, int_to_label=int_to_label)

    # Sample from the model
    sample_conditonal(conditional_model, n_classes, 1000, int_to_label=int_to_label)

