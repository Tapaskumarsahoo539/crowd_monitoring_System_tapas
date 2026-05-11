import cv2
import argparse
import os
from ultralytics import YOLO

def monitor_crowd(source, model_path, output_dir, threshold=9, conf_threshold=0.5):
    """
    Runs YOLO11 inference on a source (image, video, or webcam),
    counts the number of people, detects overcrowding, and saves the output.
    
    Args:
        source: Path to image/video or '0' for webcam.
        model_path: Path to the trained YOLO11 weights (e.g., best.pt).
        output_dir: Directory to save the output.
        threshold: Overcrowding threshold.
    """
    print(f"Loading model from {model_path}...")
    # Load the YOLO model
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"\n[ERROR] Failed to load model weights: {model_path}")
        print(f"Details: {e}")
        print("\nIf you are trying to use a newly released model like YOLO11, your 'ultralytics' library might not support it yet, or the weights file couldn't be automatically downloaded.")
        print("Please ensure:")
        print("1. You have the latest ultralytics package: pip install -U ultralytics")
        print("2. Or manually download the .pt weights file and place it in the project folder.\n")
        return
    
    # Check if source is webcam
    is_webcam = source == '0'
    if is_webcam:
        # Using cv2.CAP_DSHOW is highly recommended on Windows for webcam stability
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source: {source}")
        return

    # Setup VideoWriter if not an image
    # Note: If it's a static image, we'll handle it differently inside the loop
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30 # Default to 30 for webcam
    
    is_image = False
    if not is_webcam:
        ext = os.path.splitext(source)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            is_image = True
    
    os.makedirs(output_dir, exist_ok=True)
    out_filename = "output_webcam.mp4" if is_webcam else f"output_{os.path.basename(source)}"
    out_path = os.path.join(output_dir, out_filename)
    
    out_writer = None
    if not is_image:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))

    print("Starting inference... Press 'q' to stop.")
    
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            if is_image:
                print("Image processed successfully.")
            else:
                if frame_count == 0 and is_webcam:
                    print("Error: Could not read from webcam. Please check if your camera is connected and not used by another application (like Zoom or Teams).")
                else:
                    print("Video stream ended.")
            break
        
        frame_count += 1
            
        # Run YOLO11 inference on the frame
        # class=0 ensures we only detect 'person' if using pretrained model
        results = model.predict(frame, classes=[0], conf=conf_threshold, verbose=False)
        
        # Count the number of people detected
        # results[0].boxes contains the bounding box predictions
        person_count = len(results[0].boxes)
        
        # Plot the bounding boxes on the frame
        annotated_frame = results[0].plot(conf=True)
        
        # Density Analysis / Overcrowding Detection
        overcrowded = person_count > threshold
        
        # Display Information on the frame
        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Background rectangle for text for better visibility
        cv2.rectangle(annotated_frame, (10, 10), (350, 120), (0, 0, 0), -1)
        
        # Put Crowd Count
        cv2.putText(annotated_frame, f"Crowd Count: {person_count}", (20, 50), 
                    font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Overcrowding Status
        status_color = (0, 0, 255) if overcrowded else (0, 255, 0) # Red if overcrowded, Green if normal
        status_text = "Status: OVERCROWDED" if overcrowded else "Status: Normal"
        cv2.putText(annotated_frame, status_text, (20, 100), 
                    font, 1, status_color, 2, cv2.LINE_AA)
        
        # Show the frame (Only if we have a display, might crash on Colab without display)
        try:
            cv2.imshow("Smart Crowd Monitoring", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except cv2.error:
            # Running in headless environment (e.g., Colab)
            pass

        # Save output
        if is_image:
            cv2.imwrite(out_path, annotated_frame)
            print(f"Output image saved to {out_path}")
            break
        else:
            if out_writer:
                out_writer.write(annotated_frame)

    # Release resources
    cap.release()
    if out_writer:
        out_writer.release()
    cv2.destroyAllWindows()
    print(f"Processing complete. Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Crowd Monitoring and Analysis")
    parser.add_argument("--source", type=str, default="0", help="Path to image/video or '0' for webcam")
    parser.add_argument("--model", type=str, default="yolo11m.pt", help="Path to trained model weights")
    parser.add_argument("--output", type=str, default="../outputs", help="Directory to save output")
    parser.add_argument("--threshold", type=int, default=9, help="Overcrowding threshold number")
    parser.add_argument("--conf", type=float, default=0.65, help="Confidence threshold for detection")
    
    args = parser.parse_args()
    
    # Convert output path relative to script if necessary
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, args.output)
    model_path = os.path.join(script_dir, '..', 'models', args.model)
    if not os.path.exists(model_path):
        # Fallback to local execution using default yolo11 downloads
        model_path = args.model
    
    monitor_crowd(args.source, model_path, out_dir, args.threshold, args.conf)
