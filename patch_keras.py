import zipfile
import json
import tempfile
import os
import shutil
import sys

# Ensure utf-8 encoding for stdout
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

project_root = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(project_root, 'ai', 'weights', 'dr_grading_model.keras')
patched_model_path = os.path.join(project_root, 'ai', 'weights', 'dr_grading_model_patched.keras')

temp_dir = tempfile.mkdtemp()

print("Extracting:", model_path)
with zipfile.ZipFile(model_path, 'r') as zip_ref:
    zip_ref.extractall(temp_dir)

config_path = os.path.join(temp_dir, 'config.json')
with open(config_path, 'r', encoding='utf-8') as f:
    config_data = f.read()

print("Patching batch_shape to batch_input_shape...")
config_data = config_data.replace('"batch_shape":', '"batch_input_shape":')

with open(config_path, 'w', encoding='utf-8') as f:
    f.write(config_data)

print("Repacking to:", patched_model_path)
with zipfile.ZipFile(patched_model_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, temp_dir)
            zipf.write(file_path, arcname)

shutil.rmtree(temp_dir)
print("Done!")
