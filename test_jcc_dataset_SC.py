
import os
import cv2
import random
import sys
from extract_roi3_4_jcc_SC import classify_occluder_two_stage

def test_on_dataset(dataset_dir, samples_per_class=5):
    """
    Test classification on random samples from each class in the dataset.
    """
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory not found: {dataset_dir}")
        return

    # Model is lazy-loaded by classify_occluder_two_stage
    
    classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    classes.sort()
    
    total_correct = 0
    total_tested = 0
    
    print(f"Testing on {samples_per_class} random images per class from: {dataset_dir}")
    print("-" * 60)
    print(f"{'True Class':<25} | {'Predicted Class':<25} | {'Result':<10} | {'Image'}")
    print("-" * 60)
    
    for class_name in classes:
        class_dir = os.path.join(dataset_dir, class_name)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not images:
            continue
            
        # Select random samples
        samples = random.sample(images, min(len(images), samples_per_class))
        
        for img_name in samples:
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Error reading {img_path}")
                continue
                
            prediction = classify_occluder_two_stage(img)
            
            # Check correctness (relaxed matching)
            is_correct = False
            if class_name == prediction:
                is_correct = True
            elif 'grey' in class_name and 'grey' in prediction:
                is_correct = True
            elif 'blue' in class_name and 'blue' in prediction and 'filled' in prediction:
                is_correct = True
            
            # Special logic for JCC power/axis refine
            # The folder names are like 'red_power_refine', 'green_axis_refine'
            # The prediction might be exactly that.
            
            result_str = "PASS" if is_correct else "FAIL"
            color_code = "\033[92m" if is_correct else "\033[91m"
            reset_code = "\033[0m"
            
            print(f"{class_name:<25} | {prediction:<25} | {color_code}{result_str}{reset_code:<10} | {img_name}")
            
            total_tested += 1
            if is_correct:
                total_correct += 1
                
    print("-" * 60)
    accuracy = (total_correct / total_tested) * 100 if total_tested > 0 else 0
    print(f"Accuracy: {total_correct}/{total_tested} ({accuracy:.1f}%)")

if __name__ == "__main__":
    dataset_dir = "jcc_occluder_dataset"
    test_on_dataset(dataset_dir)
