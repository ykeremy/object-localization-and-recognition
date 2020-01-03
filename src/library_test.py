import torch
from torchvision import transforms, datasets

from src.helpers import get_label
from src.preprocessing import Resize, Padding, Normalize

data_transform = transforms.Compose(
    [
        Resize(desired_dimension=224),
        transforms.ToTensor(),
        Padding(output_size=[224, 224]),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
    ]
)

image_net_dataset = datasets.ImageFolder(root="../data/train", transform=data_transform)

dataset_loader = torch.utils.data.DataLoader(
    image_net_dataset, batch_size=1, shuffle=False
)

dataset_iter = iter(dataset_loader)
batch = next(dataset_iter)

from matplotlib import pyplot as plt

to_pil = transforms.ToPILImage()

im = batch[0][0]
im = to_pil(im)
plt.figure()
plt.imshow(im)
plt.show()

print("Label:", get_label(batch[1]))
