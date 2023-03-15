import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import logging
import argparse

class MyMNISTDataset(Dataset):
    def __init__(self, npz_file, transform=None):
        data = np.load(npz_file)
        self.images = data['images']
        self.labels = data['labels']
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# 設定超參數


# 檢查 CUDA 是否可用
#use_cuda = torch.cuda.is_available()
#device = torch.device('cuda' if use_cuda else 'cpu')
#print('Using device:', device)


# 定義模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 64, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
def test(model,device):
    model = model.to(device)
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        for i in range(len(labels)):
            true_label = labels[i].item()
            predicted_label = predicted[i].item()
            logging.info(f"Image {i + 1}: True label = {true_label}, Predicted label = {predicted_label}")
        logging.info(f'Test Accuracy: {100 * correct / total:.8f}%')
def train(model,device):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 訓練模型
    model.train()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.10f}')
    logging.info("Save model.state_dict() to {}".format(args.weights))
    torch.save(model.state_dict(), args.weights)
    

def main(model,device):
    model = model.to(device)
    
    # 損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 訓練模型
    model.train()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            if (i + 1) % 100 == 0:
                logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.8f}')
    
    # 評估模型
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
             
        print(f'Test Accuracy: {100 * correct / total:.10f}%')
    logging.info("Save model.state_dict() to {}".format(args.weights))
    torch.save(model.state_dict(), args.weights)
if __name__ == "__main__":
    file_name = os.path.splitext(os.path.basename(__file__))[0]
    logging.basicConfig(
    format='%(asctime)s.%(msecs)03d %(levelname)-2s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO,
    handlers=[
        logging.FileHandler(f'{file_name}.log'),
        logging.StreamHandler()
    ]
)

    #sys.argv = [__file__,"--device_type","cuda","--weights","fine_tune.weights","--num-epochs","1000","--learning-rate","0.00001"]
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", help="Select device, during using multiple GPU",default=0,type=int)
    parser.add_argument("-ag","--allgpu",help="Using All GPU for this run",action="store_true")
    parser.add_argument("-dt", "--device_type", help="Select device",default="cuda")
    parser.add_argument("-w", "--weights", help="Custom weights file",default="mnist_cnn.state_dict")
    parser.add_argument("-m", "--mode", help="Train or test mode")
    parser.add_argument('--training-data', type=str, default=None, help='Path to the training data')
    parser.add_argument('--testing-data', type=str, default=None, help='Path to the testing data')
    parser.add_argument('--batch-size', type=int, default=600, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate for the optimizer')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of epochs to train the model')

    args = parser.parse_args()
    args_dict = vars(args)
    logging.info(args_dict)
    model = CNN()
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs

    if args.training_data is not None:
        logging.info("Loading custom train dataset:{}".format(args.training_data))
        train_dataset = MyMNISTDataset(args.training_data, transform=torchvision.transforms.ToTensor())
    if args.testing_data is not None:
        logging.info("Loading custom test dataset:{}".format(args.testing_data))
        test_dataset = MyMNISTDataset(args.testing_data, transform=torchvision.transforms.ToTensor())
    
    if args.training_data is None:
        logging.info("Loading default train dataset")
        train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    if args.testing_data is None:
        logging.info("Loading default test dataset")
        test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    #test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if args.device_type == "cuda" and torch.cuda.is_available():
        device="cuda"
        logging.info("CUDA is selected")
    elif args.device_type == "cpu":
        device="cpu"
        logging.info("CPU is selected")


    if args.weights != '' and os.path.exists(args.weights):
        logging.info("Pytorch weights file: {} exist, loading now...".format(args.weights))
        loaded_model = CNN().to(device)
        model_weights = torch.load(args.weights)
        new_model_weights = {k.replace("module.", ""): v for k, v in model_weights.items()}
        loaded_model.load_state_dict(new_model_weights)
        model = loaded_model
    elif args.weights != '' and not os.path.exists(args.weights):
        logging.info("Pytorch weights is not exist, will save to {}".format(args.weights))
    elif args.weights == '':
        logging.info("Pytorch weights file is set empty, will not save weights file")

    if args.allgpu and torch.cuda.device_count() > 1:
        logging.info(f'Using {torch.cuda.device_count()} GPUs')
        model = nn.DataParallel(model)
        logging.info("Will using {} {} devices for compute".format(torch.cuda.device_count(),device))
    else:
        logging.info("Will using {} device for compute".format(device))

    if args.mode=="train":
        train(model,device)
    elif args.mode=="test":
       test(model,device)
    elif args.mode=="all":
       main(model,device)
