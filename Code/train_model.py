import torch
import torch.nn as nn
from tqdm import tqdm
from data_loader import get_dataloaders
from finetune_classifier import CNNClassifier
from utils import set_seed

def train_model():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, class_names = get_dataloaders()
    model = CNNClassifier(num_classes=len(class_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 11):
        model.train()
        total_loss, total_correct = 0, 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_correct += (output.argmax(1) == labels).sum().item()

        acc = total_correct / len(train_loader.dataset)
        print(f"Epoch {epoch}: Loss={total_loss:.4f}, Accuracy={acc:.4f}")

    torch.save(model.state_dict(), "final_model.pth")
