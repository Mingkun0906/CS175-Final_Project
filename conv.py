import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd



# A CNN for facial expression recognition
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 256 * 8 * 8)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Function to load and preprocess data
def load_and_preprocess_data(img_dir):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = datasets.ImageFolder(img_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, test_loader

# Training function
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

    avg_loss = total_loss / num_batches
    print(f"Train Epoch: {epoch} - Average Loss: {avg_loss:.6f}")
    
# Testing function
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
    return accuracy

def main():
    # Check if CUDA is available and use it if possible
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Set the path to image directory
    img_dir = 'emotion_origins'
    train_loader, test_loader = load_and_preprocess_data(img_dir)
    
    model = EmotionCNN(num_classes=7).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    
    accuracies = {'Epoch': [], 'Accuracy': []}
    for epoch in range(1, 21):
        train(model, device, train_loader, optimizer, epoch)
        accuracy = test(model, device, test_loader)
        scheduler.step()
        
        accuracies['Epoch'].append(epoch)
        accuracies['Accuracy'].append(accuracy)
    
    # Save accuracies to Excel file
    df = pd.DataFrame(accuracies).round({'Accuracy': 4})
    df.to_excel('conv_accuracies.xlsx', index=False)
    
    # Save the trained model
    torch.save(model.state_dict(), "conv_cnn_model.pth")

    # Reload the model for testing or fine-tuning
    model.load_state_dict(torch.load("conv_cnn_model.pth"))
    test(model, device, test_loader)

if __name__ == "__main__":
    main()
