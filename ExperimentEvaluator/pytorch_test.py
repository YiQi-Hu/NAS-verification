import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import os


# 定义网络结构
class NasNet(nn.Module):
    def __init__(self):
        super(NasNet, self).__init__()
        self.conv0 = nn.Sequential(  # input_size=(1*28*28)
            nn.Conv2d(3, 64, 3),  # padding=2保证输入输出尺寸相同
            nn.ReLU(),  # input_size=(6*28*28)
            nn.BatchNorm2d(64)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),  # input_size=(16*10*10)
            nn.BatchNorm2d(32)
        )
        self.conv2 = nn.Sequential(  # input_size=(1*28*28)
            nn.Conv2d(32, 64, 3),  # padding=2保证输入输出尺寸相同
            nn.ReLU(),  # input_size=(6*28*28)
            nn.BatchNorm2d(64)
        )
        self.conv3 = nn.Sequential(  # input_size=(1*28*28)
            nn.Conv2d(64, 128, 3),  # padding=2保证输入输出尺寸相同
            nn.ReLU(),  # input_size=(6*28*28)
            nn.BatchNorm2d(128)
        )
        self.conv4 = nn.Sequential(  # input_size=(1*28*28)
            nn.Conv2d(128, 128, 3),  # padding=2保证输入输出尺寸相同
            nn.ReLU(),  # input_size=(6*28*28)
            nn.BatchNorm2d(128)
        )
        self.conv5 = nn.Sequential(  # input_size=(1*28*28)
            nn.Conv2d(128, 192, 3),  # padding=2保证输入输出尺寸相同
            nn.ReLU(),  # input_size=(6*28*28)
            nn.BatchNorm2d(192)
        )
        self.conv6 = nn.Sequential(  # input_size=(1*28*28)
            nn.Conv2d(192, 256, 3),  # padding=2保证输入输出尺寸相同
            nn.ReLU(),  # input_size=(6*28*28)
            nn.BatchNorm2d(256)
        )
        self.conv7 = nn.Sequential(  # input_size=(1*28*28)
            nn.Conv2d(256, 256, 3),  # padding=2保证输入输出尺寸相同
            nn.ReLU(),  # input_size=(6*28*28)
            nn.BatchNorm2d(256)
        )
        self.maxpool = nn.MaxPool2d(2, 2)

    # 定义前向传播过程，输入为x
    def forward(self, x):
        output0 = self.conv0(x)
        output1 = self.conv1(output0)
        input1 = torch.cat((output0, output1), 1)
        output2 = self.conv2(input1)
        output2 = self.maxpool(output2)

        output3 = self.conv3(output2)
        output4 = self.conv4(output3)
        input2 = torch.cat((output3, output4), -1)
        output5 = self.conv5(input2)
        output5 = self.maxpool(output5)

        output6 = self.conv6(output5)
        output7 = self.conv7(output6)
        output7 = self.maxpool(output7)

        output7 = self.conv7(output7)
        output7 = self.conv7(output7)
        output7 = self.maxpool(output7)

        return output7


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        print(
            'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
            test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = NasNet()
# net = ResNet18()
# net = PreActResNet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

for epoch in range(start_epoch, start_epoch + 200):
    train(epoch)
    test(epoch)
