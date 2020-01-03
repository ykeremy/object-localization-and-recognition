import torch
from torchvision import transforms, datasets

from src.helpers import get_label

data_transform = transforms.Compose(
    [
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ]
)

image_net_dataset = datasets.ImageFolder(root="../data/train", transform=data_transform)

dataset_loader = torch.utils.data.DataLoader(
    image_net_dataset, batch_size=1, shuffle=True
)

dataset_iter = iter(dataset_loader)
batch = next(dataset_iter)

from matplotlib import pyplot as plt

to_pil = transforms.ToPILImage()

im = batch[0][0]
im = to_pil(im)
# im = np.resize(im, (224, 224, 3))
# im = im.view(224, 224, -1)
# print(im.size())
plt.figure()
plt.imshow(im)
plt.show()

print("Label:", get_label(batch[1]))
