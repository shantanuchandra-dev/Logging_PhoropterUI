import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import json
from pathlib import Path

def verify_model(model_path="chart_classifier.pth", mapping_path="class_mapping.json", data_dir="total_final_charts/val"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load mapping
    with open(mapping_path, "r") as f:
        class_mapping = json.load(f)
    
    # Invert mapping: class_name -> index
    inv_mapping = {v: int(k) for k, v in class_mapping.items()}
    num_classes = len(class_mapping)
    
    # Load model
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    results = {"correct": 0, "total": 0, "errors": []}
    
    # Walk through data_dir
    for root, dirs, files in os.walk(data_dir):
        if any(part.startswith('.') for part in Path(root).parts):
            continue
            
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = Path(root) / file
                
                # Class name is simply the directory name since we are using cleaned data
                expected_class = Path(root).name
                
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_t = transform(img).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        outputs = model(img_t)
                        _, preds = torch.max(outputs, 1)
                        predicted_class = class_mapping[str(preds.item())]
                    
                    results["total"] += 1
                    if predicted_class == expected_class:
                        results["correct"] += 1
                    else:
                        results["errors"].append({
                            "file": str(img_path),
                            "expected": expected_class,
                            "predicted": predicted_class
                        })
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

    accuracy = (results["correct"] / results["total"] * 100) if results["total"] > 0 else 0
    print(f"\nVerification Results:")
    print(f"Total Images: {results['total']}")
    print(f"Correct: {results['correct']}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    if results["errors"]:
        print(f"\nErrors ({len(results['errors'])}):")
        for error in results["errors"]:
            print(f"  File: {error['file']}")
            print(f"    Expected: {error['expected']}")
            print(f"    Predicted: {error['predicted']}")
            
    return results

if __name__ == "__main__":
    verify_model()
