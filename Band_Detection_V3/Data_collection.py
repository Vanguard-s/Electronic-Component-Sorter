import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
import sys
import os

def get_color_fingerprint(image, mask):
    """
    Calculates a 32-bin histogram (fingerprint) for the masked region.
    Fingerprint = [16 H bins] + [8 S bins] + [8 V bins]
    """
    try:
        # Get all pixels
        mask_bgr = cv2.merge([mask, mask, mask])
        masked_image = cv2.bitwise_and(image, mask_bgr)
        
        # Convert the *entire patch* to HSV
        hsv_patch = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
        
        # Create a mask to ignore black background pixels
        mask_8bit = (mask > 0).astype(np.uint8)
        
        if np.sum(mask_8bit) == 0:
            return None

        # --- Calculate Histograms ---
        # 1. Hue (16 bins)
        h_hist = cv2.calcHist([hsv_patch], [0], mask_8bit, [16], [0, 180])
        cv2.normalize(h_hist, h_hist) # Normalize to 0-1
        
        # 2. Saturation (8 bins)
        s_hist = cv2.calcHist([hsv_patch], [1], mask_8bit, [8], [0, 256])
        cv2.normalize(s_hist, s_hist)
        
        # 3. Value (8 bins)
        v_hist = cv2.calcHist([hsv_patch], [2], mask_8bit, [8], [0, 256])
        cv2.normalize(v_hist, v_hist)
        
        # --- Create 32-element fingerprint ---
        # Flatten and combine
        fingerprint = np.concatenate((h_hist.flatten(), s_hist.flatten(), v_hist.flatten()))
        
        return fingerprint

    except Exception as e:
        print(f"[Error in get_color_fingerprint]: {e}")
        return None

# --- Main Data Collection Logic ---
def collect_data(model_path, images_folder, output_csv):
    
    model = YOLO(model_path)
    color_keymap = {
        'k': 'Black', 'b': 'Brown', 'r': 'Red', 'o': 'Orange',
        'y': 'Yellow', 'g': 'Green', 'l': 'Blue', 'v': 'Violet',
        'a': 'Gray', 'w': 'White', 'd': 'Gold', 's': 'Silver'
    }
    
    if not os.path.exists(output_csv):
        # Create a header with 32 'fp' columns + 1 label column
        header = ",".join([f"fp_{i}" for i in range(32)]) + ",color_name\n"
        with open(output_csv, 'w') as f:
            f.write(header)

    for image_name in os.listdir(images_folder):
        image_path = os.path.join(images_folder, image_name)
        if not (image_name.endswith('.jpg') or image_name.endswith('.png')):
            continue
            
        image = cv2.imread(image_path)
        if image is None: continue
            
        print(f"\n--- Processing: {image_name} ---")
        results = model(image, conf=0.1)
        
        if not results[0].masks:
            print("No bands found.")
            continue
            
        masks = results[0].masks.data.cpu().numpy()
        boxes = results[0].boxes.data.cpu().numpy() 
        
        for i in range(len(masks)):
            mask_raw = masks[i]
            x1, y1, x2, y2 = boxes[i][:4]
            
            display_image = image.copy()
            cv2.rectangle(display_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            mask_8bit = (mask_raw * 255).astype(np.uint8)
            mask_resized = cv2.resize(mask_8bit, (image.shape[1], image.shape[0]))
            
            # --- Get the new fingerprint ---
            fingerprint = get_color_fingerprint(image, mask_resized)
            
            if fingerprint is None:
                continue

            cv2.imshow("Label this band", display_image)
            
            print(f"\nFound band. Fingerprint (first 4): {fingerprint[:4]}...")
            print("What color is this? (k=Black, b=Brown, r=Red, o=Orange, y=Yellow, g=Green, l=Blue, v=Violet, a=Gray, w=White, d=Gold, s=Silver)")
            print("Press 'q' to skip this band, 'n' for next image.")
            
            key = cv2.waitKey(0) & 0xFF
            key_char = chr(key)
            
            if key_char == 'q': continue
            if key_char == 'n': break
                
            if key_char in color_keymap:
                color_name = color_keymap[key_char]
                print(f"Labeling as: {color_name}")
                
                # --- Save fingerprint to CSV ---
                fp_string = ",".join([str(val) for val in fingerprint])
                with open(output_csv, 'a') as f:
                    f.write(f"{fp_string},{color_name}\n")
            else:
                print("Unknown key. Skipping band.")
                
        cv2.destroyAllWindows()

if __name__ == "__main__":
    MODEL_PATH = r'D:/Hackathon/Band_Detection_V3/ResistorProject/run_heavy_aug/weights/best.pt'
    IMAGES_FOLDER = r"C:\Users\kppse\OneDrive\Desktop\Test_1"
    OUTPUT_CSV = 'color_dataset_histogram_1.csv' # New CSV name
    
    collect_data(MODEL_PATH, IMAGES_FOLDER, OUTPUT_CSV)
    cv2.destroyAllWindows()