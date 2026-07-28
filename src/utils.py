import json
from pathlib import Path
from src.config import PROCESSED_DIR

def ensure_processed_dir(dataset_name):
    """Creates a specific folder inside data/processed/ for the current dataset"""
    processed_path = PROCESSED_DIR / dataset_name
    processed_path.mkdir(parents=True, exist_ok=True)
    return processed_path

def save_metadata(metadata, dataset_name="default_dataset"):
    processed_path = ensure_processed_dir(dataset_name)
    output_path = processed_path / "metadata.json"
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    return output_path

def load_metadata(metadata_path):
    metadata_path = Path(metadata_path)
    
    if not metadata_path.exists():
        # Fallback check inside data/processed
        processed_fallback = PROCESSED_DIR / metadata_path.name
        if processed_fallback.exists():
            metadata_path = processed_fallback
        else:
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")
            
    with open(metadata_path, 'r') as f:
        return json.load(f)

def get_unique_classes_counts(metadata):
    unique_classes = set()
    count_options = {}

    for item in metadata:
        if 'class_counts' in item:
            for cls_name, count in item['class_counts'].items():
                unique_classes.add(cls_name)
                
                if cls_name not in count_options:
                    count_options[cls_name] = set()
                    
                count_options[cls_name].add(count)

    unique_classes = sorted(list(unique_classes))
    for cls in count_options:
        count_options[cls] = sorted(list(count_options[cls]))
    
    return unique_classes, count_options