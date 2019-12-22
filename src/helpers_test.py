from random import randint

import matplotlib.pyplot as plt

from src.helpers import ImageNetDataset

imagenet_dataset = ImageNetDataset('../data/train')

print('Number of images in the dataset: {}', len(imagenet_dataset))
img_idx = randint(0, len(imagenet_dataset))

fig = plt.figure()
image = imagenet_dataset[img_idx]
plt.imshow(image)
plt.show()

