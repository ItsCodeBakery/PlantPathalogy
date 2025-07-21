import torch
from data_loader import get_dataloaders
from finetune_classifier import CNNClassifier

def evaluate_model():
    _, test_loader, class_names = get_dataloaders()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNClassifier(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load("final_model.pth"))
    model.eval()

    correct = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()

    print(f"Test Accuracy: {correct / len(test_loader.dataset):.4f}")
