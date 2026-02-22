import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from app import MODEL_PATHS
    print("SUCCESS: app.py imported successfully.")
except Exception as e:
    print(f"ERROR: Failed to import app.py: {e}")
    sys.exit(1)

# Verify Model Paths
all_models_exist = True
for name, path in MODEL_PATHS.items():
    full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', path))
    if os.path.exists(full_path):
        print(f"OK: Model {name} found at {full_path}")
    else:
        print(f"FAIL: Model {name} NOT found at {full_path}")
        all_models_exist = False

# Verify Datasets
datasets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets'))
required_datasets = ['rgb', 'grey', 'hsv']
all_datasets_exist = True
if os.path.exists(datasets_dir):
    print(f"OK: Datasets directory found at {datasets_dir}")
    for ds in required_datasets:
        ds_path = os.path.join(datasets_dir, ds)
        if os.path.exists(ds_path) and os.listdir(ds_path):
            print(f"OK: Dataset {ds} found and not empty.")
        else:
            print(f"FAIL: Dataset {ds} missing or empty at {ds_path}")
            all_datasets_exist = False
else:
    print(f"FAIL: Datasets directory missing at {datasets_dir}")
    all_datasets_exist = False

if all_models_exist and all_datasets_exist:
    print("VERIFICATION PASSED: All checks successful.")
    sys.exit(0)
else:
    print("VERIFICATION FAILED: Missing files.")
    sys.exit(1)
