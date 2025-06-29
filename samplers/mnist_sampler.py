import torch
import random
from torchvision import datasets, transforms

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