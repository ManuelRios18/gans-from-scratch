import os
import copy
import tqdm
import torch
from torch import nn
import torch.optim as optim
from samplers.mnist_sampler import MNISTImagesSampler
from utils import reset_dir, save_grid, plot_learning_curves

Z_DIM = 100

class Discriminator(nn.Module):

    def __init__(self,
                 input_image_size: tuple=(1, 28, 28)) -> None:
        # This Convolutional architecture is simple a sequence of
        # convolutional layers followed by a fully connected layer that
        # performs the classification. This is a classical LeNet like
        # architecture without MaxPooling layers as they are replaced with
        # convolutions with stride.
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        flattened_size = self._get_flattened_size(input_image_size)
        self.linear_layers = nn.Sequential(
            nn.Flatten(),  # Flatten the output of conv_layers
            nn.Linear(in_features=flattened_size, out_features=32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=32, out_features=1)
        )

    def _get_flattened_size(self, input_image_size: tuple) -> int:
        """Method to calculate the size of the flattened volume"""
        dummy_input = torch.randn(1, *input_image_size)
        temp_model = copy.deepcopy(self.conv_layers)

        with torch.no_grad():
            output = temp_model(dummy_input)

        return output.numel()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        conv_output = self.conv_layers(image)
        prediction = self.linear_layers(conv_output)
        return prediction

class Generator(nn.Module):
    def __init__(self) -> None:
        # This is a fully deconvolutional architecture the transform 1x1
        # images with Z_DIM channels into 28x28 single channel images.
        # To calculate ConvTranspose2d parameters the utility function
        # compute_conv_transposed_output_size was used.
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=Z_DIM,
                out_channels=32,
                kernel_size=7,
                stride=1
            ),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=16,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.Tanh()
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        image = self.convolutions(z)
        return image

def sample_z(n_samples: int) -> torch.Tensor:
    sampled_z = torch.normal(mean=0, std=1, size=(n_samples, Z_DIM))
    return sampled_z.view(n_samples, Z_DIM, 1, 1)


def main() -> None:
    x_sampler = MNISTImagesSampler()
    discriminator = Discriminator()
    generator = Generator()

    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
    else:
        device = "cpu"

    print(f"Using device: {device}")

    logs_path = "training_logs/dcgan"
    reset_dir(logs_path)

    discriminator.to(device)
    generator.to(device)

    # Generate some initial z to plot while training.
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

        # Detach tensors to avoid gradient accumulation on the generator
        # params.
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
            print(f"Iteration {iteration}: "
                  f"D_loss -> {discriminator_loss.item():.4f},"
                  f" G_loss -> {generator_loss.item():.4f}")
            with torch.no_grad():
                sampled_images = generator(z_to_plot).detach().cpu().numpy()
            save_grid(images=sampled_images,
                      title=f"DCGAN generation {iteration}",
                      file_name=os.path.join(logs_path, f"{iteration}.png"))


    final_generations = generator(z_to_plot).detach().cpu().numpy()
    save_grid(images=final_generations,
              title="DCGAN generation",
              file_name=os.path.join(logs_path, "post_training.png"))
    plot_learning_curves(d_losses=d_losses,
                         g_losses=g_losses,
                         title="Learning Curves DCGAN",
                         file_name=os.path.join(logs_path, "curves.png"))



main()