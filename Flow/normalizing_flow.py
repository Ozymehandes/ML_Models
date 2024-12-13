import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
from model import NormalizingFlowModel
from create_data import create_unconditional_olympic_rings

DATA_POINTS = 250000
VALIDATION_SPLIT = 0.05
SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_loader, val_loader, optimizer, scheduler, n_epochs, normal_dist):
    train_losses = []
    val_losses = []

    val_det = []
    val_log_prob = []
    for epoch in range(n_epochs):
        model.train()
        ep_train_loss = []
        

        for batch, in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            
            z = model.inverse(batch)
            log_prob_z = normal_dist.log_prob(z)
            log_det_jacobian = model.log_det_jacobian(batch)
            loss = torch.mean(-log_prob_z - log_det_jacobian)
            loss.backward()
            optimizer.step()

            ep_train_loss.append(loss.item())

        scheduler.step()
        train_losses.append(np.mean(ep_train_loss))

        # Validation step
        model.eval()
        ep_val_loss = []
        ep_val_det = []
        ep_val_log_prob = []
        with torch.no_grad():
            for batch, in val_loader:
                batch = batch.to(DEVICE)
                z = model.inverse(batch)
                log_prob_z = normal_dist.log_prob(z)
                log_det_jacobian = model.log_det_jacobian(batch)
                loss = torch.mean(-log_prob_z - log_det_jacobian)
                ep_val_loss.append(loss.item())
                ep_val_det.append(log_det_jacobian.mean().item())
                ep_val_log_prob.append(log_prob_z.mean().item())
        
        val_losses.append(np.mean(ep_val_loss))
        val_det.append(np.mean(ep_val_det))
        val_log_prob.append(np.mean(ep_val_log_prob))
        print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {np.mean(ep_train_loss):.4f}, Val Loss: {np.mean(ep_val_loss):.4f}')

    # Generate the ticks
    ticks = np.arange(1, n_epochs+1, 1)

    # plt.plot(range(1, n_epochs+1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, n_epochs+1), val_losses, label='Loss', marker='o')
    plt.plot(range(1, n_epochs+1), val_det, label='Log Det', marker='o')
    plt.plot(range(1, n_epochs+1), val_log_prob, label='Log Prob', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Q3.3.1 - Normalizing Flow Losses, Log Prob and Det over Epochs')
    plt.legend()
    
    # Set the x-axis ticks
    plt.xticks(ticks)
    plt.savefig('NormalizingFlow_loss_plot.png')
    plt.close()
    return model

def sample_trajectory(model, n_samples):
    model.eval()
    normal_dist = torch.distributions.MultivariateNormal(torch.zeros(model.input_dim).to(DEVICE),
                                                        torch.eye(model.input_dim).to(DEVICE))
    z = normal_dist.sample((n_samples,)).to(DEVICE)
    x = [[z[i, 0].detach().cpu().numpy()] for i in range(n_samples)]
    y = [[z[i, 1].detach().cpu().numpy()] for i in range(n_samples)]
    for i, layer in enumerate(t_model.normalizing_flow):
        z = layer(z)
        if i % 2 == 0:
            for j in range(n_samples):
                x[j].append(z[j, 0].detach().cpu().numpy())
                y[j].append(z[j, 1].detach().cpu().numpy())

    return x, y, len(x[0])

if __name__ == '__main__':
    # Set random seed for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Prepare data
    data = create_unconditional_olympic_rings(DATA_POINTS, verbose=False)
    data_shape = data.shape
    data = torch.tensor(data, dtype=torch.float32)
    train_set, val_set = train_test_split(data, test_size=VALIDATION_SPLIT, random_state=SEED)

    # Create data loaders
    train_loader = DataLoader(TensorDataset(train_set), batch_size=128, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_set), batch_size=128, shuffle=False)

    # Define normal distribution
    normal_dist = torch.distributions.MultivariateNormal(torch.zeros(data_shape[1]).to(DEVICE), 
                                                         torch.eye(data_shape[1]).to(DEVICE))

    # Initialize the normalizing flow model and learning parameters
    model = NormalizingFlowModel(data_shape[1], affine_layers_n=15).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_epochs = 20
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Train the model and plot the losses
    trained_model = train_model(model, train_loader, val_loader, optimizer, scheduler, n_epochs, normal_dist)

    # Save the trained model
    torch.save(trained_model.state_dict(), 'trained_normalizing_flow.pth')

    t_model = NormalizingFlowModel(data_shape[1], affine_layers_n=15).to(DEVICE)
    t_model.load_state_dict(torch.load('trained_normalizing_flow.pth'))
    t_model.eval()

    # Sample from the model with different seeds
    n_samples = 1000
    seeds = [5, 23, 42]

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        z = normal_dist.sample((n_samples,)).to(DEVICE)
        x = t_model(z)
        x = x.detach().cpu().numpy()
        plt.scatter(x[:, 0], x[:, 1])
        plt.title(f'Q3.3.2 - Samples from the model with seed {seed}')
        plt.savefig(f'NormalizingFlow_samples_seed_{seed}.png')
        plt.close()

    # Sampling over time(layers)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    z = normal_dist.sample((n_samples,)).to(DEVICE)
    layer_indices = [0, 5, 11, 17, 23, 28]

    for idx, layer in enumerate(t_model.normalizing_flow):
        z = layer(z)
        if idx in layer_indices:
            x = z.detach().cpu().numpy()
            plt.scatter(x[:, 0], x[:, 1])
            plt.title(f'Q3.3.3 - Samples from the model after {idx + 1} layers')
            plt.savefig(f'NormalizingFlow_samples_layer_{idx + 1}.png')
            plt.close()
    

    # Samples over time
    n_samples = 10
    z = normal_dist.sample((n_samples,)).to(DEVICE)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    x, y, t_values = sample_trajectory(t_model, 10)
    t_values = np.arange(t_values)
    # Adjust the figure size and subplot parameters
    plt.figure(figsize=(16, 10))
    plt.subplots_adjust(left=0.3)

    lines = []  # List to store the line objects
    labels = []  # List to store the line labels
    for i in range(n_samples):
        plt.scatter(x[i], y[i], c=t_values)
        # Connect the points with a line
        line, = plt.plot(x[i], y[i], linewidth=0.5)
        lines.append(line)
        labels.append(f'Point {i + 1}')
        # Mark the starting point with a red 'x'
        start = plt.scatter(x[i][0], y[i][0], color='red', marker='x')
        # Mark the ending point with a green 'o'
        end = plt.scatter(x[i][-1], y[i][-1], color='green', marker='x')

    # Add the start and end markers to the lines and labels lists
    lines.extend([start, end])
    labels.extend(['Start', 'End'])

    plt.title('Q3.3.4 - Trajectories of points through the model')
    plt.colorbar(label='Time t')
    # Create a combined legend for the lines and markers, and place it to the left of the plot
    plt.legend(lines, labels, bbox_to_anchor=(-0.2, 1), loc='upper left')
    plt.savefig('NormalizingFlow_trajectories.png')
    plt.close()

    # Reverse trajectories
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    point_in = np.array([[0, 0.5], [1.5, 0.5], [-0.5, -1]])
    point_out = np.array([[-1.5, -1.5], [2, -1]])
    points = np.concatenate([point_in, point_out], axis=0)
    points = torch.tensor(points, dtype=torch.float32).to(DEVICE)
    n_samples = points.shape[0]

    model.eval()
    x = [[points[i, 0].detach().cpu().numpy()] for i in range(n_samples)]
    y = [[points[i, 1].detach().cpu().numpy()] for i in range(n_samples)]
    for i, layer in enumerate(reversed(t_model.normalizing_flow)):
        points = layer.inverse(points)
        if i % 2 == 0:
            for j in range(n_samples):
                x[j].append(points[j, 0].detach().cpu().numpy())
                y[j].append(points[j, 1].detach().cpu().numpy())

    t_values = len(x[0])
    t_values = np.arange(t_values)



    # Adjust the figure size and subplot parameters
    plt.figure(figsize=(10, 6))

    # Plot for the first three points
    plt.subplot(1, 2, 1)
    lines = []  # List to store the line objects
    labels = []  # List to store the line labels
    for i in range(3):
        plt.scatter(x[i], y[i], c=t_values)
        # Connect the points with a line
        line, = plt.plot(x[i], y[i], linewidth=0.5)
        lines.append(line)
        labels.append(f'Point {i + 1}')
        # Mark the starting point with a red 'x'
        start = plt.scatter(x[i][0], y[i][0], color='red', marker='x')
        # Mark the ending point with a green 'o'
        end = plt.scatter(x[i][-1], y[i][-1], color='green', marker='x')
    plt.title('Q3.3.5 - Reverse Trajectories of points inside the rings')

    # Plot for the remaining points
    plt.subplot(1, 2, 2)
    for i in range(3, n_samples):
        plt.scatter(x[i], y[i], c=t_values)
        # Connect the points with a line
        line, = plt.plot(x[i], y[i], linewidth=0.5)
        lines.append(line)
        labels.append(f'Point {i + 1}')
        # Mark the starting point with a red 'x'
        start = plt.scatter(x[i][0], y[i][0], color='red', marker='x')
        # Mark the ending point with a green 'o'
        end = plt.scatter(x[i][-1], y[i][-1], color='green', marker='x')
    plt.title('Q3.3.5 - Reverse Trajectories of points the rings')

    # Add the start and end markers to the lines and labels lists
    lines.extend([start, end])
    labels.extend(['Start', 'End'])

    # Create the legend outside of the plots
    plt.legend(lines, labels, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig('ReverseTrajectories.png')
    plt.close()

    log_prob = normal_dist.log_prob(points)
    print(f'Q3.3.5 - Log probability of the points:\n {log_prob}')