import nbformat as nbf

nb = nbf.v4.new_notebook()

# Cell 1: Description
md1 = '''# Tree Species Classification (Module 1) - Full 19 Species Retraining
This notebook is fully automated to train the EfficientNet-B3 model on all 19 tree species on Kaggle.'''
nb.cells.append(nbf.v4.new_markdown_cell(md1))

# Cell 2: Setup Codebase
code_setup = """# 1. Automatically find where Kaggle put the codebase zip and extract it
import os, glob

zip_files = glob.glob('/kaggle/input/**/*.zip', recursive=True)
target = [z for z in zip_files if 'tree_species' in z]

if not target:
    print('Checking if codebase was already unpacked by Kaggle...')
    req_files = glob.glob('/kaggle/input/**/requirements.txt', recursive=True)
    if req_files:
        code_dir = os.path.dirname(req_files[0])
        os.system(f"cp -r '{code_dir}'/* /kaggle/working/")
    else:
        print('ERROR: Could not find your codebase files.')
else:
    zip_path = target[0]
    print(f'Found ZIP at: {zip_path}')
    !unzip -q "{zip_path}" -d .

print('Installing requirements...')
!pip install -q -r requirements.txt
"""
nb.cells.append(nbf.v4.new_code_cell(code_setup))

# Cell 3: Map Datasets
code_data = """# 2. Map Kaggle Datasets into data/raw/ so the pipeline sees all 19 species
import os

print('Linking Datasets into the codebase...')
os.makedirs('data/raw/plant_leaves', exist_ok=True)
os.makedirs('data/raw/medicinal_leaves', exist_ok=True)

# Find where Kaggle mounted the Plant Leaves dataset and copy its contents
os.system("cp -r /kaggle/input/plant-leaves-for-image-classification/*/* data/raw/plant_leaves/ 2>/dev/null || true")
os.system("cp -r /kaggle/input/plant-leaves-for-image-classification/* data/raw/plant_leaves/ 2>/dev/null || true")

# Find where Kaggle mounted the Medicinal datasets and copy contents
os.system("cp -r /kaggle/input/indian-medicinal-leaves-dataset/*/* data/raw/medicinal_leaves/ 2>/dev/null || true")
os.system("cp -r /kaggle/input/indian-medicinal-leaves-dataset/* data/raw/medicinal_leaves/ 2>/dev/null || true")

print('Files mapped successfully! You are ready to process all 19 species.')
"""
nb.cells.append(nbf.v4.new_code_cell(code_data))

# Cell 4: Train pipeline
code_train = """# 3. Process, Split, and Train!
!python data/prepare_dataset.py
!python data/make_splits.py
!python training/train.py
"""
nb.cells.append(nbf.v4.new_code_cell(code_train))

# Cell 5: Evaluate
code_eval = """# 4. Generate Final Metrics for the Teacher (Accuracy, F1, Confusion Matrix)
print('=============================================')
print('Running Evaluation on Test Set...')
!python evaluation/evaluate.py --checkpoint outputs/checkpoints/best.pt

print('Moving Confusion Matrix and metrics over to Kaggle outputs!')
!cp outputs/plots/test_confusion_matrix.png /kaggle/working/ 2>/dev/null || true
!cp outputs/plots/test_confusion_matrix_normalized.png /kaggle/working/ 2>/dev/null || true
!cp outputs/logs/test_metrics.json /kaggle/working/ 2>/dev/null || true
!cp outputs/checkpoints/best.pt /kaggle/working/ 2>/dev/null || true
"""
nb.cells.append(nbf.v4.new_code_cell(code_eval))

# Cell 6: Inference
code_inf = """# 5. Manual Testing / Inference
import torch
import torchvision.transforms as T
from PIL import Image
import glob
import matplotlib.pyplot as plt

test_images = glob.glob('data/processed/*/*.jpg')
if test_images:
    sample_path = test_images[0]  
    print(f'Testing on sample: {sample_path}')
    
    img = Image.open(sample_path).convert('RGB')
    
    transform = T.Compose([
        T.Resize((300, 300)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        from training.model import TreeSpeciesModel
        import yaml
        
        with open('configs/config.yaml', 'r') as f:
            cfg = yaml.safe_load(f)
            
        model = TreeSpeciesModel(cfg['model']['backbone'], num_classes=len(cfg['class_map']))
        checkpoint = torch.load('outputs/checkpoints/best.pt', map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        model.to(input_tensor.device)
        model.eval()
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, dim=0)
            
        class_names = list(cfg['class_map'].keys())
        class_names.sort()  
        
        try:
            predicted_class = class_names[predicted_idx.item()]
        except:
            predicted_class = f'Class ID {predicted_idx.item()}'
            
        plt.imshow(img)
        plt.title(f'Prediction: {predicted_class} ({confidence.item()*100:.2f}%)')
        plt.axis('off')
        plt.show()
    except Exception as e:
        print('Inference failed:', e)
"""
nb.cells.append(nbf.v4.new_code_cell(code_inf))

with open('d:/Capston Zip Files/tree_carbon_assessment_system/module1_species_classification/Kaggle_Module1_Training.ipynb', 'w') as f:
    nbf.write(nb, f)
print('New Notebook generated successfully!')
