import os
import json
import torch
from torchvision import transforms
from PIL import Image

# Load class mapping
with open("class_mapping.json", "r") as f:
    class_mapping = json.load(f)

# Invert mapping for easy lookup
idx_to_class = {int(k): v for k, v in class_mapping.items()}
class_to_idx = {v: int(k) for k, v in class_mapping.items()}

# Load model
from torchvision import models
import torch.nn as nn
model = models.resnet18()
num_classes = len(class_mapping)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load("chart_classifier.pth", map_location=torch.device('cpu')))
model.eval()

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(img_path):
    img = Image.open(img_path).convert('RGB')
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        conf, pred = torch.max(probabilities, 0)
        return idx_to_class[pred.item()], conf.item()

def test_all_the_charts(root_dir):
    results = []
    for class_folder in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_folder)
        if not os.path.isdir(class_path):
            continue
        # Normalize folder name to match mapping
        folder_name = class_folder.strip().lower().replace(' ', '_').replace('.', '').replace('<', '').replace('>', '')
        for img_file in os.listdir(class_path):
            if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            img_path = os.path.join(class_path, img_file)
            pred_class, conf = predict_image(img_path)
            correct = pred_class == folder_name
            results.append((img_path, folder_name, pred_class, conf, correct))
    # Print summary
    total = len(results)
    correct = sum(1 for r in results if r[-1])
    print(f"Accuracy: {correct}/{total} ({100.0*correct/total:.2f}%)")
    for r in results:
        print(f"Image: {r[0]} | True: {r[1]} | Pred: {r[2]} | Conf: {r[3]:.2f} | {'CORRECT' if r[4] else 'WRONG'}")

if __name__ == "__main__":
    test_all_the_charts("all_the_charts")
