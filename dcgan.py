import copy
import tqdm
import torch
import random
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision import datasets, transforms

#TODO: Change generator architecture

Z_DIM = 100

class Discriminator(nn.Module):

    def __init__(self, input_image_size=(1, 28, 28)):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        flattened_size = self._get_flattened_size(input_image_size)
        self.linear_layers = nn.Sequential(
            nn.Flatten(),  # Flatten the output of conv_layers
            nn.Linear(in_features=flattened_size, out_features=32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=32, out_features=1)
        )

    def _get_flattened_size(self, input_image_size):
        x = torch.randn(1, *input_image_size)
        temp_model = copy.deepcopy(self.conv_layers)

        with torch.no_grad():
            output = temp_model(x)

        return output.numel()

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.linear_layers(x)
        return x

class Generator(nn.Module):
    def __init__(self):
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

    def forward(self, z):
        x = self.linear_stack(z)
        return x.view(-1, 1, 28, 28)


class MNISTImagesSampler:
    def __init__(self):
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

    def sample(self, n_samples):
        if self.position + n_samples > len(self.indices):
            random.shuffle(self.indices)
            self.position = 0

        batch_indices = self.indices[self.position:self.position + n_samples]
        self.position += n_samples

        batch = [self.dataset[i] for i in batch_indices]
        data, _ = zip(*batch)
        return torch.stack(data)

def sample_z(n_samples: int):
    return torch.normal(mean=0, std=1, size=(n_samples, Z_DIM))


x_sampler = MNISTImagesSampler()
discriminator = Discriminator()
generator = Generator()

if torch.accelerator.is_available():
    device = torch.accelerator.current_accelerator()
else:
    device = "cpu"

print(f"Using device: {device}")

discriminator.to(device)
generator.to(device)

# Generate some initial data to see how an untrained model behaves.
z_to_plot = sample_z(n_samples=16).to(device)
real_data_sample = x_sampler.sample(n_samples=16).to(device)
pre_training_generations = generator.forward(z_to_plot).detach().cpu().numpy()


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

    # Detach tensors to avoid gradient accumulation on the generator params.
    generated_x = generator.forward(z).detach()

    d_real_pred = discriminator.forward(x)
    d_fake_pred = discriminator.forward(generated_x)

    real_loss = bce_loss(d_real_pred, torch.ones(batch_size, 1).to(device))
    fake_loss = bce_loss(d_fake_pred, torch.zeros(batch_size, 1).to(device))
    discriminator_loss = (real_loss + fake_loss) / 2

    discriminator_loss.backward()
    optimizer_d.step()

    # Optimize the generator
    optimizer_g.zero_grad()
    z = sample_z(n_samples=batch_size).to(device)
    generated_x = generator.forward(z)
    d_fake_pred = discriminator.forward(generated_x)
    # According to Goodfellow et al. Minimizing log(1-D(G(z))) can cause
    # saturation. Therefore, authors suggest to maximize D(G(z)).
    # This suggestion is equivalent to minimize -D(G(z)) as implemented below
    generator_loss = bce_loss(d_fake_pred,
                              torch.ones(batch_size, 1).to(device))

    generator_loss.backward()
    optimizer_g.step()
    d_losses.append(discriminator_loss.item())
    g_losses.append(generator_loss.item())
    if iteration % 1000 == 0:
        print(f"Epoch {iteration}: D_loss -> {discriminator_loss.item():.4f},"
              f" G_loss -> {generator_loss.item():.4f}")

post_training_generations = generator.forward(z_to_plot).detach().cpu().numpy()

plt.figure()
plt.title("Training Losses")
plt.plot(d_losses, label="Discriminator Loss")
plt.plot(g_losses, label="Generator Loss")
plt.legend()
plt.grid()


def show_grid(imgs_np, title):

    imgs = torch.tensor(imgs_np, dtype=torch.float32)
    grid = make_grid(imgs, nrow=4, padding=2, normalize=True, value_range=(-1, 1))
    plt.figure(figsize=(4,4))
    plt.title(title)
    # grid is (C, H, W); move channels last for imshow
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    plt.axis('off')


show_grid(real_data_sample,
          title="Real data ")
show_grid(pre_training_generations,
          title="Generator output  —  before training")
show_grid(post_training_generations,
          title="Generator output  —  after training")
plt.show()
