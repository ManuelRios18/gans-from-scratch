import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt


class Discriminator(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(in_features=2, out_features=20),
            nn.ReLU(),
            nn.Linear(in_features=20, out_features=20),
            nn.ReLU(),
            nn.Linear(in_features=20, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_stack(x)


class Generator(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(in_features=5, out_features=20),
            nn.ReLU(),
            nn.Linear(in_features=20, out_features=20),
            nn.ReLU(),
            nn.Linear(in_features=20, out_features=2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear_stack(z)

def sample_z(n_samples: int) -> torch.Tensor:
    return torch.normal(mean=0, std=1, size=(n_samples, 5))

def sample_x(n_samples: int) -> torch.Tensor:
    return torch.normal(mean=2, std=0.5, size=(n_samples, 2))


def main() -> None:
    discriminator = Discriminator()
    generator = Generator()

    # Generate some initial data to see how an untrained model behaves.
    z_to_plot = sample_z(n_samples=100)
    real_data_sample = sample_x(n_samples=100)
    with torch.no_grad:
        pre_training_generations = generator(z_to_plot).detach().numpy()

    batch_size = 128
    num_iterations = 50000
    learning_rate = 0.00005
    optimizer_g = optim.Adam(params=generator.parameters(),
                             lr=learning_rate,
                             weight_decay=0)
    optimizer_d = optim.Adam(params=discriminator.parameters(),
                             lr=learning_rate,
                             weight_decay=0)
    bce_loss = nn.BCELoss()
    g_losses, d_losses = [], []
    for iteration in range(num_iterations):
        # Optimize the discriminator
        optimizer_d.zero_grad()
        z = sample_z(n_samples=batch_size)
        x = sample_x(n_samples=batch_size)
        # Detach tensors to avoid gradient accumulation on the generator params.
        generated_x = generator(z).detach()

        d_real_pred = discriminator(x)
        d_fake_pred = discriminator(generated_x)

        real_loss = bce_loss(d_real_pred, torch.ones(batch_size, 1))
        fake_loss = bce_loss(d_fake_pred, torch.zeros(batch_size, 1))
        discriminator_loss = (real_loss + fake_loss) / 2

        discriminator_loss.backward()
        optimizer_d.step()

        # Optimize the generator
        optimizer_g.zero_grad()
        z = sample_z(n_samples=batch_size)
        generated_x = generator(z)
        d_fake_pred = discriminator(generated_x)
        # According to Goodfellow et al. Minimizing log(1-D(G(z))) can cause
        # saturation. Therefore, authors suggest to maximize D(G(z)).
        # This suggestion is equivalent to minimize -D(G(z)) as implemented below
        generator_loss = bce_loss(d_fake_pred, torch.ones(batch_size, 1))

        generator_loss.backward()
        optimizer_g.step()
        d_losses.append(discriminator_loss.item())
        g_losses.append(generator_loss.item())
        print(f"Iterations {iteration}: "
              f"D_loss -> {discriminator_loss.item():.4f},"
              f" G_loss -> {generator_loss.item():.4f}")

    post_training_generations = generator(z_to_plot).detach().numpy()

    plt.figure()
    plt.title("Training Losses")
    plt.plot(d_losses, label="Discriminator Loss")
    plt.plot(g_losses, label="Generator Loss")
    plt.legend()
    plt.grid()
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Generator Output Comparison', fontsize=16)

    # Pre-Training Plot
    axes[0].set_title("Pre-Training")
    axes[0].scatter(real_data_sample[:, 0],
                    real_data_sample[:, 1],
                    label='Real Data',
                    alpha=0.6,
                    c='blue')
    axes[0].scatter(pre_training_generations[:, 0],
                    pre_training_generations[:, 1],
                    label='Generated Data',
                    alpha=0.6,
                    c='red')
    axes[0].set_xlabel("X-axis")
    axes[0].set_ylabel("Y-axis")
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_aspect('equal', adjustable='box')

    axes[1].set_title("Post-Training")
    axes[1].scatter(real_data_sample[:, 0],
                    real_data_sample[:, 1],
                    label='Real Data',
                    alpha=0.6,
                    c='blue')
    axes[1].scatter(post_training_generations[:, 0],
                    post_training_generations[:, 1],
                    label='Generated Data',
                    alpha=0.6,
                    c='red')
    axes[1].set_xlabel("X-axis")
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_aspect('equal', adjustable='box')

    plt.show()

main()
