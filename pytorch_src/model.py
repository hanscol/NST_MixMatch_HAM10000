import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNN, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)

        self.pool = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout2d(p=0.25)

        self.fc1 = nn.Linear(9216, 128)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc2 = nn.Linear(128, out_channels)
        self.soft = nn.Softmax()

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(self.relu(x))
        x = self.drop1(self.pool(self.relu(x)))

        x = x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])

        x = self.drop2(self.relu(self.fc1(x)))
        x = self.fc2(x)
        x = self.soft(x)

        return x