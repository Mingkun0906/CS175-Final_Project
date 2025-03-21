import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt  # Added for plotting


class EmotionResNet(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        
        for param in list(self.resnet.parameters())[:-20]:
            param.requires_grad = False
            
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.resnet(x)
        return F.log_softmax(x, dim=1)


def load_and_preprocess_data(img_dir):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(img_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    avg_loss = total_loss / num_batches
    print(f"Train Epoch: {epoch} - Average Loss: {avg_loss:.6f}")
    return avg_loss  # Return avg_loss so we can store it


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.0f}%)\n')
    return accuracy, test_loss  # Return both accuracy and test_loss


# Added function to generate graphs
def generate_training_graphs(epochs, accuracies, losses):
    plt.figure(figsize=(16, 7))
    
    # Training Accuracy Graph
    plt.subplot(1, 2, 1)
    plt.plot(epochs, accuracies, marker='o', linestyle='-', color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("ResNet Training Accuracy")
    plt.grid(True, alpha=0.3)
    plt.xticks(epochs)
    
    # Training Loss Graph
    plt.subplot(1, 2, 2)
    plt.plot(epochs, losses, marker='o', linestyle='-', color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ResNet Training Loss")
    plt.grid(True, alpha=0.3)
    plt.xticks(epochs)
    
    plt.tight_layout()
    plt.savefig('resnet_training_metrics.png', dpi=300)
    plt.show()


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    img_dir = 'emotion_origins'
    train_loader, test_loader = load_and_preprocess_data(img_dir)
    model = EmotionResNet(num_classes=7).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00005)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.8)
    
    # Modified to track both accuracy and loss
    metrics = {'Epoch': [], 'Accuracy': [], 'Loss': []}
    
    for epoch in range(1, 11):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        accuracy, test_loss = test(model, device, test_loader)
        scheduler.step()
        
        metrics['Epoch'].append(epoch)
        metrics['Accuracy'].append(accuracy)
        metrics['Loss'].append(train_loss)  # Store training loss
    
    # Save metrics to Excel
    df = pd.DataFrame(metrics).round({'Accuracy': 2, 'Loss': 6})
    df.to_excel('resnet50_metrics.xlsx', index=False)
    
    # Generate training graphs
    generate_training_graphs(metrics['Epoch'], metrics['Accuracy'], metrics['Loss'])
    
    torch.save(model.state_dict(), "resnet50_model.pth")
    
    model.load_state_dict(torch.load("resnet50_model.pth"))
    test(model, device, test_loader)


if __name__ == "__main__":
    main()