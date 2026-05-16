import argparse
import random
import wandb
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision.transforms import v2
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Argument Parser Setup
parser = argparse.ArgumentParser(description="Train a simple CNN on CIFAR-10 based on the PyTorch tutorial with WanDB integration")
parser.add_argument("--project_name", type=str, default="OMA-CIFAR10", help="WanDB project name")
parser.add_argument("--run_name", type=str, default="run", help="WanDB run name")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
parser.add_argument("--dropout_rate", type=float, default=0.0, help="Dropout rate for fully connected layers")
parser.add_argument("--conv_ch1", type=int, default=6, help="Number of channels for conv layer 1")
parser.add_argument("--conv_ch2", type=int, default=16, help="Number of channels for conv layer 2")
parser.add_argument("--fc1_dim", type=int, default=120, help="Dimension of first fully connected layer")
parser.add_argument("--fc2_dim", type=int, default=84, help="Dimension of second fully connected layer")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer")
parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for optimizer")
parser.add_argument("--epochs", type=int, default=2, help="Number of epochs to train")
parser.add_argument("--save_model", type=bool, default=False, help="Whether to save the trained model")
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

class Net(nn.Module):
    def __init__(self, dropout_rate=0.0, conv_ch1=6, conv_ch2=16, fc1_dim=120, fc2_dim=84):
        super().__init__()
        self.conv1 = nn.Conv2d(3, conv_ch1, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(conv_ch1, conv_ch2, 5)
        self.fc1 = nn.Linear(conv_ch2 * 5 * 5, fc1_dim)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.fc3 = nn.Linear(fc2_dim, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
def show_random_images(classes, trainloader):
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(args.batch_size)))
    
def test(model, dataloader):
    correct = 0
    total = 0
    model.eval() # Turn off dropout
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train() # Turn dropout back on
    return 100 * correct / total

def main():
    # 3. Initialize Weights & Biases
    wandb.init(project=args.project_name, name=args.run_name, config=args)

    # 4. Data Loading & Preprocessing
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./dataset', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=2)
    
    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # show_random_images(classes, trainloader)

    # 5. Model, Criterion, and Optimizer
    # Initialize Model (Tutorial CNN)
    model = Net(dropout_rate=args.dropout_rate, conv_ch1=args.conv_ch1, conv_ch2=args.conv_ch2, fc1_dim=args.fc1_dim, fc2_dim=args.fc2_dim)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(DEVICE)
    
    # Define Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # 6. Training Loop
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        model.train() # Set model to training mode (important for dropout)
        running_loss = 0.0
        epoch_train_correct = 0
        epoch_train_total = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # --- Metrics Accumulation ---
            running_loss += loss.item()
            with torch.no_grad():
                _, predicted = torch.max(outputs, 1)
                epoch_train_total += labels.size(0)
                epoch_train_correct += (predicted == labels).sum().item()
            
            # Optional: Live Progress (Keep this for your own sanity during long runs)
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] Batch Loss: {loss.item():.3f}')
                wandb.log({"batch_loss": loss.item(),})
                
        # --- End of Epoch Calculations ---
        # 1. Calculate TRUE Epoch Training Metrics
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100 * epoch_train_correct / epoch_train_total
        
        # 2. Run Evaluation on Test Set
        # Note: test() should internally handle model.eval() and model.train()
        test_accuracy = test(model, testloader)

        # 3. Formal Logging (WandB and Print)
        print(f'--- Epoch {epoch + 1} Summary ---')
        print(f'Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | Test Acc: {test_accuracy:.2f}%')
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "train_acc": epoch_acc,
            "test_acc": test_accuracy
        })
                
    print('Finished Training')
    if args.save_model:
        torch.save(model.state_dict(), "cifar10_net.pth")

    # 7. Validation Loop (After Finishing Training)
    test(model, testloader)
    wandb.log({"final_val_accuracy": test(model, testloader)})

if __name__ == "__main__":
    main()
