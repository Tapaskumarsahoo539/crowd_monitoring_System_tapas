import os
from ultralytics import YOLO

def train_model(data_yaml_path, epochs=50, batch_size=16, imgsz=640, model_type='yolov8n.pt'):
    """
    Trains a YOLOv8 model for crowd detection.
    
    Args:
        data_yaml_path: Path to the data.yaml file.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        imgsz: Image size for training.
        model_type: Which YOLOv8 model to start with ('yolov8n.pt' or 'yolov8s.pt').
    """
    print(f"Loading base model {model_type}...")
    model = YOLO(model_type)
    
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
    
    # We use YOLOv8n (nano) as requested for lightweight detection
    train_model(yaml_path, epochs=30, batch_size=16, model_type='yolov8n.pt')
