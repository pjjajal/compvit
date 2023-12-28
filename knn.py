import torch
import torchvision.transforms as tvt
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from tqdm import tqdm
from dinov2.factory import dinov2_factory

TRANSFORM = tvt.Compose(
    [
        tvt.RandomCrop(32, padding=4),
        tvt.Resize(112),
        tvt.RandomHorizontalFlip(),
        tvt.ToTensor(),
        tvt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

model, _ = dinov2_factory('dinov2_vits14')
model.load_state_dict(torch.load("dinov2/checkpoints/dinov2_vits14_pretrain.pth"))
model = model.to(device="cuda")

train_dataset = CIFAR100("toy_experiments/data",train=False, transform=TRANSFORM, download=True)
train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=8,
)


embeddings = torch.tensor([])
for data, _ in tqdm(train_loader):
    data = data.to("cuda")
    with torch.no_grad():
        outputs = model(data, is_training=True)
        embedding = model.norm(outputs['x_prenorm'])
        B, N, C = embedding.shape
        embedding = embedding.view(B, -1).cpu()
        embeddings = torch.cat([embeddings, embedding], dim=0)

torch.save(embeddings, "cifar100_dino_embeddings_train.pt")