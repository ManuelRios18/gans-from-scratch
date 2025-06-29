import os
import tqdm
import torch
import random
from torch import nn
import torch.optim as optim
from torchvision import datasets, transforms
from utils import reset_dir, save_grid, plot_learning_curves

Z_DIM = 100

class Discriminator(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=28*28, out_features=128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=128, out_features=64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=64, out_features=32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=32, out_features=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_stack(x)


class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(in_features=Z_DIM, out_features=32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=32, out_features=64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=64, out_features=128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=128, out_features=28 * 28),
            nn.Tanh()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.linear_stack(z)
        return x.view(-1, 1, 28, 28)


class MNISTImagesSampler:
    def __init__(self) -> None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, ),std=(0.5, ))
        ])

        self.dataset = datasets.MNIST(root='./data',
                                       train=True,
                                       download=True,
                                       transform=transform)
        self.indices = list(range(len(self.dataset)))
        random.shuffle(self.indices)
        self.position = 0

    def sample(self, n_samples: int) -> torch.Tensor:
        if self.position + n_samples > len(self.indices):
            random.shuffle(self.indices)
            self.position = 0

        batch_indices = self.indices[self.position:self.position + n_samples]
        self.position += n_samples

        batch = [self.dataset[i] for i in batch_indices]
        data, _ = zip(*batch)
        return torch.stack(data)

def sample_z(n_samples: int) -> torch.Tensor:
    return torch.normal(mean=0, std=1, size=(n_samples, Z_DIM))


def main() -> None:
    x_sampler = MNISTImagesSampler()
    discriminator = Discriminator()
    generator = Generator()
    device = "cpu"
    logs_path = "training_logs/gan"
    reset_dir(logs_path)

    discriminator.to(device)
    generator.to(device)

    # Generate some initial data to see how an untrained model behaves.
    z_to_plot = sample_z(n_samples=16).to(device)

    batch_size = 64
    num_iterations = 93750
    learning_rate = 0.0002
    optimizer_g = optim.Adam(params=generator.parameters(),
                             lr=learning_rate,
                             weight_decay=0)
    optimizer_d = optim.Adam(params=discriminator.parameters(),
                             lr=learning_rate,
                             weight_decay=0)
    bce_loss = nn.BCEWithLogitsLoss()
    g_losses, d_losses = [], []
    for iteration in tqdm.tqdm(range(num_iterations)):
        # Optimize the discriminator
        optimizer_d.zero_grad()
        z = sample_z(n_samples=batch_size).to(device)
        x = x_sampler.sample(n_samples=batch_size).to(device)

        # Detach tensors to avoid gradient accumulation.
        generated_x = generator(z).detach()

        real_predictions = discriminator(x)
        fake_predictions = discriminator(generated_x)

        real_loss = bce_loss(real_predictions,
                             torch.ones_like(real_predictions).to(device))
        fake_loss = bce_loss(fake_predictions,
                             torch.zeros_like(fake_predictions).to(device))
        discriminator_loss = (real_loss + fake_loss) / 2

        discriminator_loss.backward()
        optimizer_d.step()

        # Optimize the generator
        optimizer_g.zero_grad()
        z = sample_z(n_samples=batch_size).to(device)
        generated_x = generator(z)
        fake_predictions = discriminator(generated_x)
        # According to Goodfellow et al. Minimizing log(1-D(G(z))) can cause
        # saturation. Therefore, authors suggest to maximize D(G(z)).
        # This suggestion is equivalent to minimize -D(G(z))
        # as implemented below
        generator_loss = bce_loss(fake_predictions,
                                  torch.ones_like(fake_predictions).to(device))

        generator_loss.backward()
        optimizer_g.step()
        d_losses.append(discriminator_loss.item())
        g_losses.append(generator_loss.item())
        if iteration % 1000 == 0:
            print(f"Epoch {iteration}: "
                  f"D_loss -> {discriminator_loss.item():.4f},"
                  f" G_loss -> {generator_loss.item():.4f}")
            with torch.no_grad():
                sampled_images = generator(z_to_plot).detach().cpu().numpy()
            save_grid(images=sampled_images,
                      title=f"GAN generation {iteration}",
                      file_name=os.path.join(logs_path, f"{iteration}.png"))

    final_generations = generator(z_to_plot).detach().cpu().numpy()
    save_grid(images=final_generations,
              title="GAN generation",
              file_name=os.path.join(logs_path, "post_training.png"))
    plot_learning_curves(d_losses=d_losses,
                         g_losses=g_losses,
                         title="Learning Curves GAN",
                         file_name=os.path.join(logs_path, "curves.png"))
main()
