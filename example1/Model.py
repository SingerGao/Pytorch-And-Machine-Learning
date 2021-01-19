import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1) # 128*128*32
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3) # 126*126*32
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1) # 63*63*64
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3) # 61*61*64
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(30*30*64, 512)
        self.fc2 = nn.Linear(512, 17)
        

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 30*30*64)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x

if __name__ == '__main__':
    net = Net()
    print(net)
