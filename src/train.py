import torch
from torch import optim, nn
from torchvision import datasets, transforms

from src.network import FeedForwardNN

PATH = 'model.pth'
NUM_EPOCHS = 10
HIDDEN_SIZE = 32
BATCH_SIZE = 16
LR = 1e-3

data_transform = transforms.Compose([
    # Resize
    # Pad
    # Normalize
    # Extract features
])
train_dataset = datasets.ImageFolder(root='../data/train',
                                         transform=data_transform)
loader_train = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=16, shuffle=True)

network = FeedForwardNN(hidden_size=HIDDEN_SIZE)
optimizer = optim.SGD(network.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

for e in range(NUM_EPOCHS):
    for b, data in enumerate(loader_train):
        X, y = data

        optimizer.zero_grad()

        y_hat = network(X)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()

        print('[{e+1}, {b+1}] loss:{loss.item()}')

print('Training complete!')
print('Saved at {MODEL_PATH}')
torch.save(network, PATH)

# network = torch.load(PATH)
