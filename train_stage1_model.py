"""
Stage 1 Model Training
Trains a binary classifier: filled vs jcc_pattern
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os
from occluder_model import OccluderNet

# Params
DATA_DIR = 'stage1_dataset'
MODEL_PATH = 'stage1_model.pth'
CLASSES_FILE = 'stage1_classes.txt'
BATCH_SIZE = 8
EPOCHS = 40
LR = 0.001

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Dataset directory {DATA_DIR} not found.")
        print("Please run prepare_stage1_data.py first.")
        return

    # Transforms (with augmentation for robustness)
    train_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load full dataset
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
    
    print(f"Classes: {full_dataset.classes}")
    print(f"Total images: {len(full_dataset)}")
    
    # Save classes for inference mapping
    with open(CLASSES_FILE, 'w') as f:
        for idx, cls_name in enumerate(full_dataset.classes):
            f.write(f"{idx}:{cls_name}\n")
    print(f"Saved class mapping to {CLASSES_FILE}")
    
    # Split dataset into train and validation (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Update validation dataset transform
    val_dataset.dataset.transform = val_transforms
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Training samples: {train_size}, Validation samples: {val_size}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Binary classification: 2 classes
    model = OccluderNet(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    best_val_acc = 0.0
    
    print("\nStarting training...")
    print("=" * 70)
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  → Saved best model (Val Acc: {val_acc:.2f}%)")
    
    print("\n" + "=" * 70)
    print("Finished Training")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
