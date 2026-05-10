import os
import glob
import cv2
import scipy.io
import numpy as np
import shutil
from tqdm import tqdm

def preprocess_ucf_qnrf(dataset_dir, output_dir, box_size=32):
    """
    Converts UCF-QNRF .mat point annotations into YOLO format bounding boxes.
    
    Args:
        dataset_dir: Path to the root dataset directory containing images and ground_truth folders.
        output_dir: Path to save the YOLO formatted dataset.
        box_size: The fixed width and height of the bounding box around each point.
    """
    print(f"Starting preprocessing of UCF-QNRF dataset from {dataset_dir}...")
    
    # Define directories
    img_dir = os.path.join(dataset_dir, 'images')
    gt_dir = os.path.join(dataset_dir, 'ground_truth')
    
    yolo_img_train_dir = os.path.join(output_dir, 'images', 'train')
    yolo_img_val_dir = os.path.join(output_dir, 'images', 'val')
    yolo_label_train_dir = os.path.join(output_dir, 'labels', 'train')
    yolo_label_val_dir = os.path.join(output_dir, 'labels', 'val')
    
    # Ensure output directories exist
    os.makedirs(yolo_img_train_dir, exist_ok=True)
    os.makedirs(yolo_img_val_dir, exist_ok=True)
    os.makedirs(yolo_label_train_dir, exist_ok=True)
    os.makedirs(yolo_label_val_dir, exist_ok=True)

    # Note: UCF-QNRF has Train and Test folders originally. 
    # For this script, we assume all images are in 'images' and we will split them.
    # If the user has them separated, they can modify this script.
    # Here we do a simple 80-20 train-val split if they are all in one folder.
    
    image_files = glob.glob(os.path.join(img_dir, '*.jpg'))
    if not image_files:
        print("No images found! Make sure dataset is extracted to dataset/images and dataset/ground_truth.")
        return

    np.random.seed(42)
    np.random.shuffle(image_files)
    
    split_idx = int(0.8 * len(image_files))
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    def process_files(files, split_name):
        out_img_dir = yolo_img_train_dir if split_name == 'train' else yolo_img_val_dir
        out_label_dir = yolo_label_train_dir if split_name == 'train' else yolo_label_val_dir
        
        print(f"Processing {split_name} split ({len(files)} files)...")
        for img_path in tqdm(files):
            img_name = os.path.basename(img_path)
            basename = os.path.splitext(img_name)[0]
            
            # Mat file naming in UCF-QNRF is usually img_XXXX_ann.mat or similar
            # Sometimes it's just exactly the same name but .mat
            # Let's assume standard UCF-QNRF format: img_XXXX_ann.mat
            mat_path = os.path.join(gt_dir, f"{basename}_ann.mat")
            if not os.path.exists(mat_path):
                # Try simple .mat extension
                mat_path = os.path.join(gt_dir, f"{basename}.mat")
                if not os.path.exists(mat_path):
                    print(f"Warning: Annotation not found for {img_name}. Skipping.")
                    continue
            
            # Read image dimensions
            img = cv2.imread(img_path)
            if img is None:
                continue
            h_img, w_img, _ = img.shape
            
            # Read MAT file
            mat = scipy.io.loadmat(mat_path)
            # The key in UCF-QNRF is usually 'annPoints'
            if 'annPoints' in mat:
                points = mat['annPoints']
            else:
                # Sometimes it's buried inside
                print(f"Warning: 'annPoints' not found in {mat_path}. Skipping.")
                continue
            
            # Copy image to YOLO directory
            shutil.copy(img_path, os.path.join(out_img_dir, img_name))
            
            # Create YOLO label file
            label_path = os.path.join(out_label_dir, f"{basename}.txt")
            with open(label_path, 'w') as f:
                for point in points:
                    x, y = point[0], point[1]
                    
                    # Create bounding box (x_center, y_center, width, height) normalized
                    x_center = min(max(x / w_img, 0.0), 1.0)
                    y_center = min(max(y / h_img, 0.0), 1.0)
                    
                    # Width and height normalized
                    w_norm = box_size / w_img
                    h_norm = box_size / h_img
                    
                    # YOLO format: class_id x_center y_center width height
                    # class_id is 0 for 'person'
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

    process_files(train_files, 'train')
    process_files(val_files, 'val')
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    # Assuming script is run from project root or inside scripts folder
    # Adjust paths accordingly
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_source = os.path.join(base_dir, 'dataset')
    dataset_output = os.path.join(base_dir, 'dataset', 'yolo_format')
    
    preprocess_ucf_qnrf(dataset_source, dataset_output, box_size=32)
