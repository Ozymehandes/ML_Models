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

def train_unconditional_model(model, train_loader, val_loader, optimizer, scheduler, n_epochs):
    criterion = torch.nn.MSELoss()
    normal_dist = torch.distributions.MultivariateNormal(torch.zeros(model.input_dim).to(DEVICE),
                                                        torch.eye(model.input_dim).to(DEVICE))
    uniform_dist = torch.distributions.Uniform(0, 1)
    train_losses = []
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = []

        for batch, in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            eps = normal_dist.sample((batch.shape[0],))
            t = uniform_dist.sample((batch.shape[0],)).unsqueeze(-1)
            y = t*batch + (1-t)*eps
            y_pred = model(y, t)
            v_t = batch - eps
            loss = criterion(y_pred, v_t)
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())

        scheduler.step()
        train_losses.append(np.mean(epoch_loss))
        print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {np.mean(epoch_loss):.4f}')
    
    # Generate the ticks
    ticks = np.arange(1, n_epochs+1, 1)

    plt.plot(range(1, n_epochs+1), train_losses, label='Train Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Q4.3.1 - Uncontional Flow-Matching Losses')
    plt.legend()
    
    # Set the x-axis ticks
    plt.xticks(ticks)
    plt.savefig('unconditionalFlow_losses.png')
    plt.close()

    return model

def sample_unconditional_for_t(model, n_samples, delta_t=1e-3):
    model.eval()
    normal_dist = torch.distributions.MultivariateNormal(torch.zeros(model.input_dim).to(DEVICE),
                                                        torch.eye(model.input_dim).to(DEVICE))

    plotted_ts = [0, 0.2, 0.4, 0.6, 0.8, 1]
    y = normal_dist.sample((n_samples,)).to(DEVICE)
    t = 0
    while t <= 1:
        t = round(t, 3)
        t_tensor = torch.tensor([t]*n_samples, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
        next = model(y, t_tensor)*delta_t
        y = y + next
        if t in plotted_ts:
            plt.figure(figsize=(16, 9))
            detached = y.detach().cpu().numpy()
            plt.scatter(detached[:, 0], detached[:, 1])
            plt.title(f'Samples at t={round(t, 1)}')
            plt.savefig(f'Q4.3.2 - samples_flow_progression_with_{t}.png')
            plt.close()
        t += delta_t

    return y

def sample_uncoditional(model, n_samples, delta_t):
    model.eval()
    normal_dist = torch.distributions.MultivariateNormal(torch.zeros(model.input_dim).to(DEVICE),
                                                        torch.eye(model.input_dim).to(DEVICE))
    y = normal_dist.sample((n_samples,)).to(DEVICE)
    t = 0
    while t <= 1:
        t = round(t, 3)
        t_tensor = torch.tensor([t]*n_samples, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
        next = model(y, t_tensor)*delta_t
        y = y + next
        t += delta_t

    plt.figure(figsize=(16, 9))
    detached = y.detach().cpu().numpy()
    plt.scatter(detached[:, 0], detached[:, 1])
    plt.title(f'Q4.3.4 - Samples with delta_t={delta_t}')
    plt.savefig(f'Q4.3.4 - samples_with_delta_t={delta_t}.png')
    plt.close()
    return y

def reverse_samples_uncoditional(model, samples, delta_t):
    model.eval()
    n_samples = samples.shape[0]
    x = [[samples[i, 0].detach().cpu().numpy()] for i in range(n_samples)]
    y = [[samples[i, 1].detach().cpu().numpy()] for i in range(n_samples)]
    t_values = [1]
    t = 1
    while t >= 0:
        t = round(t, 3)
        t_tensor = torch.tensor([t]*n_samples, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
        next = model(samples, t_tensor)*delta_t
        samples = samples - next
        for i in range(n_samples):
            x[i].append(samples[i, 0].detach().cpu().numpy())
            y[i].append(samples[i, 1].detach().cpu().numpy())
        t_values.append(t)
        t -= delta_t

    return x, y, t_values

def sample_trajectory(model, n_samples):
    model.eval()
    normal_dist = torch.distributions.MultivariateNormal(torch.zeros(model.input_dim).to(DEVICE),
                                                        torch.eye(model.input_dim).to(DEVICE))
    delta_t = 1e-3
    samples = normal_dist.sample((n_samples,)).to(DEVICE)
    x = [[samples[i, 0].detach().cpu().numpy()] for i in range(n_samples)]
    y = [[samples[i, 1].detach().cpu().numpy()] for i in range(n_samples)]
    t_values = [0]
    t = 0
    while t <= 1:
        t = round(t, 3)
        t_tensor = torch.tensor([t]*n_samples, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
        next = model(samples, t_tensor)*delta_t
        samples = samples + next
        for i in range(n_samples):
            x[i].append(samples[i, 0].detach().cpu().numpy())
            y[i].append(samples[i, 1].detach().cpu().numpy())
        t += delta_t
        t_values.append(t)


    return x, y, t_values


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
    model = UnconditionalFlow(data_shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_epochs = 20
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Train the model
    trained_model = train_unconditional_model(model, train_loader, val_loader, optimizer, scheduler, n_epochs)

    # Save the trained model
    torch.save(trained_model.state_dict(), 'trained_unconditional_flow.pth')

    t_model = UnconditionalFlow(data_shape[1]).to(DEVICE)
    t_model.load_state_dict(torch.load('trained_unconditional_flow.pth'))

    # Sample from the trained model for different values of t
    n_samples = 1000
    sampled_points = sample_unconditional_for_t(t_model, n_samples)
    
    # Sample 10 and plot trajectories
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    x, y, t_values = sample_trajectory(t_model, 10)

    for i in range(10):
        plt.scatter(x[i], y[i], c=t_values)

        # Mark the starting point with a red 'x'
        start = plt.scatter(x[i][0], y[i][0], color='red', marker='x')
        # Mark the ending point with a green 'o'
        end = plt.scatter(x[i][-1], y[i][-1], color='green', marker='o')

    plt.title('Trajectories of 10 sampled points')
    plt.colorbar(label='Time t')
    plt.legend([start, end], ['Start', 'End'])
    plt.savefig('Q4.3.3 - Sample trajectories.png')
    plt.close()


    # Sample from the trained model for different values of delta_t
    
    delta_ts = [0.002,0.02,0.05,0.1,0.2]
    for delta_t in delta_ts:
        sample_uncoditional(t_model, n_samples, delta_t)
    

    # Reversing the flow
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    point_in = np.array([[0, 0.5], [1.5, 0.5], [-0.5, -1]])
    point_out = np.array([[-1.5, -1.5], [2, -1]])
    points = np.concatenate([point_in, point_out], axis=0)
    points = torch.tensor(points, dtype=torch.float32).to(DEVICE)

    points_tensor = torch.tensor(points, dtype=torch.float32).to(DEVICE)
    print("Original points: \n", points_tensor)
    x, y, t_values = reverse_samples_uncoditional(t_model, points_tensor, 1e-3)
    for i in range(5):
        plt.scatter(x[i], y[i], c=t_values)

        if i >= 3:
            # Draw a circle around the starting point
            circle = plt.Circle((x[i][0], y[i][0]), 0.1, color='black', fill=False)
            plt.gca().add_patch(circle)

        start = plt.scatter(x[i][0], y[i][0], color='red', marker='x')
        # Mark the ending point with a green 'o'
        end = plt.scatter(x[i][-1], y[i][-1], color='green', marker='o')

    plt.title('Q4.3.5 - Reverse Trajectories of 5 sampled points')
    plt.colorbar(label='Time t')
    plt.legend([start, end], ['Start', 'End'])
    plt.savefig('Q4.3.5 - reverse Sample trajectories.png')
    plt.close()

    reversed_points = np.array([(x[i][-1], y[i][-1]) for i in range(5)])
    reveresed_points_tensor = torch.tensor(reversed_points, dtype=torch.float32).to(DEVICE)
    
    delta_t = 1e-3
    y = reveresed_points_tensor
    t = 0
    while t <= 1:
        t = round(t, 3)
        t_tensor = torch.tensor([t]*5, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
        next = model(y, t_tensor)*delta_t
        y = y + next
        t += delta_t
    
    print("Points after reverse and forward: \n", y)
