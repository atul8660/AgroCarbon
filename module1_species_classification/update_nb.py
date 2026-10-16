import nbformat as nbf

filepath = r'd:/Capston Zip Files/tree_carbon_assessment_system/module1_species_classification/Kaggle_Module1_Training.ipynb'
with open(filepath, 'r') as f:
    nb = nbf.read(f, as_version=4)

dynamic_unzip = """# Automatically find where Kaggle put the codebase zip and extract it
import os, glob

zip_files = glob.glob('/kaggle/input/**/*.zip', recursive=True)
target = [z for z in zip_files if 'tree_species' in z]

if not target:
    print('ERROR: Could not find tree_species_classifier_linux.zip in /kaggle/input/')
    print('Did you upload the zip successfully? It should be in the right sidebar.')
else:
    zip_path = target[0]
    print(f'Found ZIP at: {zip_path}')
    !unzip -q "{zip_path}" -d tree_species_classifier/
    %cd tree_species_classifier
    !pip install -r requirements.txt
"""

for cell in nb.cells:
    if 'unzip' in cell.source and 'tree_species' in cell.source:
        cell.source = dynamic_unzip

with open(filepath, 'w') as f:
    nbf.write(nb, f)
print('Notebook updated!')
