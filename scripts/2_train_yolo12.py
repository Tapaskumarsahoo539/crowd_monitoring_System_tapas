import os
from ultralytics import YOLO

def train_model(data_yaml_path, epochs=50, batch_size=16, imgsz=640, model_type='yolo12n.pt'):
    """
    Trains a YOLO12 model for crowd detection.
    
    Args:
        data_yaml_path: Path to the data.yaml file.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        imgsz: Image size for training.
        model_type: Which YOLO12 model to start with ('yolo12n.pt' or 'yolo12s.pt').
    """
    print(f"Loading base model {model_type}...")
    try:
        model = YOLO(model_type)
    except Exception as e:
        print(f"\n[ERROR] Failed to load model weights: {model_type}")
        print(f"Details: {e}")
        print("\nIf you are trying to use a newly released model like YOLO12, your 'ultralytics' library might not support it yet.")
        print("Please ensure:")
        print("1. You have the latest ultralytics package: pip install -U ultralytics")
        print("2. Or manually download the .pt weights file and place it in the project folder.\n")
        return
    
    print(f"Starting training on {data_yaml_path} for {epochs} epochs...")
    # Train the model
    # Results are automatically saved to runs/detect/train by default
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        patience=10,        # Early stopping if no improvement
        project='runs/train', # Directory to save training results
        name='crowd_model',   # Name of the current run
        device='',          # Automatically selects GPU if available, else CPU
    )
    
    print("Training complete!")
    print(f"Best model weights are saved in: runs/train/crowd_model/weights/best.pt")

if __name__ == "__main__":
    # Path to data.yaml
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yaml_path = os.path.join(base_dir, 'data.yaml')
    
    # We use yolo12n (nano) as requested for lightweight detection
    train_model(yaml_path, epochs=30, batch_size=16, model_type='yolo12n.pt')
