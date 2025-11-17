import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
import sys
import joblib 
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import os

# ==============================================================================
# 1. COLOR & RESISTOR VALUE DEFINITIONS
# ==============================================================================
# (All maps - DIGIT_MAP, MULTIPLIER_MAP, TOLERANCE_MAP, TCR_MAP - are unchanged)
DIGIT_MAP = {
    'Black': 0, 'Brown': 1, 'Red': 2, 'Orange': 3, 'Yellow': 4,
    'Green': 5, 'Blue': 6, 'Violet': 7, 'Gray': 8, 'White': 9
}
MULTIPLIER_MAP = {
    'Black': 10**0, 'Brown': 10**1, 'Red': 10**2, 'Orange': 10**3,
    'Yellow': 10**4, 'Green': 10**5, 'Blue': 10**6, 'Violet': 10**7,
    'Gray': 10**8, 'White': 10**9, 'Gold': 10**-1, 'Silver': 10**-2
}
TOLERANCE_MAP = {
    'Brown': 1, 'Red': 2, 'Green': 0.5, 'Blue': 0.25,
    'Violet': 0.1, 'Gray': 0.05, 'Gold': 5, 'Silver': 10
}
TCR_MAP = { 
    'Brown': 100, 'Red': 50, 'Orange': 15, 'Yellow': 25,
    'Blue': 10, 'Violet': 5, 'Black': 250, 'White': 1
}
# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================

class ColorClassifierDNN:
    def __init__(self, model_path, assets_path):
        """Loads the pre-trained DNN histogram model and encoder."""
        try:
            # Load the .keras model
            self.model = tf.keras.models.load_model(model_path)
            
            # Load the assets file (which contains the encoder)
            assets = joblib.load(assets_path)
            self.encoder = assets['encoder']
            print("Successfully loaded DNN histogram model and encoder.")

        except FileNotFoundError as e:
            print(f"Error: Could not find a required file. {e}")
            self.model = None
        except Exception as e:
            print(f"An error occurred loading the models: {e}")
            self.model = None
            
    def get_color_name(self, fingerprint):
        """Predicts the color name from a 32-element histogram fingerprint."""
        if self.model is None or fingerprint is None:
            return "Unknown"
            
        # Reshape for the model (1 sample, 32 features)
        fingerprint_array = fingerprint.reshape(1, 32)
        
        # Predict probabilities
        prediction_probabilities = self.model.predict(fingerprint_array, verbose=0)
        
        # Get the index of the highest probability
        predicted_index = np.argmax(prediction_probabilities, axis=1)[0]
        
        # Use the encoder to turn the index back to a string
        color_name = self.encoder.inverse_transform([predicted_index])[0]
        
        return color_name

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
        h_hist = cv2.calcHist([hsv_patch], [0], mask_8bit, [16], [0, 180])
        cv2.normalize(h_hist, h_hist)
        
        s_hist = cv2.calcHist([hsv_patch], [1], mask_8bit, [8], [0, 256])
        cv2.normalize(s_hist, s_hist)
        
        v_hist = cv2.calcHist([hsv_patch], [2], mask_8bit, [8], [0, 256])
        cv2.normalize(v_hist, v_hist)
        
        # --- Create 32-element fingerprint ---
        fingerprint = np.concatenate((h_hist.flatten(), s_hist.flatten(), v_hist.flatten()))
        return fingerprint

    except Exception as e:
        print(f"[Error in get_color_fingerprint]: {e}")
        return None

# --- This is the robust "waterfall" decoder, unchanged ---
def decode_resistor(band_info_list):
    band_count = len(band_info_list)
    if band_count < 4 or band_count > 6:
        colors_found = [b[1] for b in band_info_list]
        return f"Error: Expected 4-6 bands, got {band_count}. (Found: {colors_found})"

    sorted_bands = sorted(band_info_list, key=lambda b: b[0])
    band_colors = [b[1] for b in sorted_bands]
    x_positions = [b[0] for b in sorted_bands]
    
    orientation_confirmed = False
    
    if band_count > 4: 
        gaps = [x_positions[i+1] - x_positions[i] for i in range(band_count - 1)]
        mean_gap = np.mean(gaps)
        largest_gap = np.max(gaps)
        largest_gap_index = np.argmax(gaps)
        if largest_gap > (mean_gap * 1.5):
            if largest_gap_index == band_count - 2:
                orientation_confirmed = True
            elif largest_gap_index == 0:
                band_colors.reverse()
                orientation_confirmed = True
    
    if not orientation_confirmed:
        if band_colors[-1] in ('Gold', 'Silver'):
            orientation_confirmed = True
        elif band_colors[0] in ('Gold', 'Silver'):
            band_colors.reverse()
            orientation_confirmed = True

    if not orientation_confirmed:
        unambiguous_digits = ['Violet', 'Blue', 'Green']
        if band_colors[-1] in unambiguous_digits:
            band_colors.reverse()
            orientation_confirmed = True

    try:
        # (Decoding logic for 4, 5, 6 bands is unchanged)
        if band_count == 4:
            digit1 = DIGIT_MAP[band_colors[0]]
            digit2 = DIGIT_MAP[band_colors[1]]
            multiplier = MULTIPLIER_MAP[band_colors[2]]
            tolerance = TOLERANCE_MAP[band_colors[3]]
            resistance = (digit1 * 10 + digit2) * multiplier
            tol_str = f"+/-{tolerance}%"
        elif band_count == 5:
            digit1 = DIGIT_MAP[band_colors[0]]
            digit2 = DIGIT_MAP[band_colors[1]]
            digit3 = DIGIT_MAP[band_colors[2]]
            multiplier = MULTIPLIER_MAP[band_colors[3]]
            tolerance = TOLERANCE_MAP[band_colors[4]]
            resistance = (digit1 * 100 + digit2 * 10 + digit3) * multiplier
            tol_str = f"+/-{tolerance}%"
        elif band_count == 6:
            digit1 = DIGIT_MAP[band_colors[0]]
            digit2 = DIGIT_MAP[band_colors[1]]
            digit3 = DIGIT_MAP[band_colors[2]]
            multiplier = MULTIPLIER_MAP[band_colors[3]]
            tolerance = TOLERANCE_MAP[band_colors[4]]
            tcr = TCR_MAP[band_colors[5]]
            resistance = (digit1 * 100 + digit2 * 10 + digit3) * multiplier
            tol_str = f"+/-{tolerance}% (TCR: {tcr} ppm/K)"

        # (Formatting logic is unchanged)
        if resistance >= 1_000_000_000:
            res_str = f"{resistance / 1_000_000_000:.1f} GOhms"
        elif resistance >= 1_000_000:
            res_str = f"{resistance / 1_000_000:.1f} MOhms"
        elif resistance >= 1_000:
            res_str = f"{resistance / 1_000:.1f} kOhms"
        else:
            res_str = f"{resistance:.1f} Ohms"
        return f"{res_str} {tol_str}"
        
    except KeyError as e:
        return f"Error: Color {e} in wrong position. (Bands: {band_colors})"
    except Exception as e:
        return f"[Error in decode_resistor]: {e}"

# ==============================================================================
# 3. MAIN PROCESSING FUNCTION (MODIFIED)
# ==============================================================================

def process_image(model_path, image_path, color_model): 
    print(f"--- 1. Loading model: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"!!! FATAL ERROR: Error loading model. Check path. Details: {e}")
        return

    print(f"--- 2. Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"!!! FATAL ERROR: Could not load image. Check path.")
        return
        
    print("--- 2b. Preprocessing image...")
    processed_image = cv2.bilateralFilter(image, 9, 75, 75)
        
    print("--- 3. Running inference...")
    try:
        # NOTE: The 'conf=0.1' here is the *initial* filter.
        # We will apply your '0.20' filter manually below.
        results = model(processed_image, conf=0.1) 
    except Exception as e:
        print(f"!!! FATAL ERROR: Error during model inference: {e}")
        return

    if not results[0].masks:
        print("--- 4. No bands were detected in the image.")
        cv2.imshow("Resistor Detection Result (No Bands Found)", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    annotated_image = results[0].plot(img=image)
    detected_bands = []
    
    try:
        masks = results[0].masks.data.cpu().numpy()
        boxes = results[0].boxes.data.cpu().numpy() 
    except Exception as e:
        print(f"!!! FATAL ERROR: Error extracting masks/boxes from results: {e}")
        return

    print(f"--- 4. Found {len(masks)} potential bands. Analyzing colors...")
    for i in range(len(masks)):
        
        # <--- ADDED: Get confidence score from the boxes array ---
        # The 'boxes' array structure is [x1, y1, x2, y2, confidence, class_id]
        confidence = boxes[i][4]
        
        # <--- ADDED: Filter to eliminate bands with confidence 20 (0.20) or below ---
        if confidence <= 0.20:
            x_center_low_conf = (boxes[i][0] + boxes[i][2]) / 2
            print(f"   - SKIPPING band at x={x_center_low_conf:.0f}: Low confidence ({confidence:.2f})")
            continue  # Skip to the next band
            
        # --- Original code continues for bands that PASS the filter ---
        mask_raw = masks[i]
        mask_8bit = (mask_raw * 255).astype(np.uint8)
        mask_resized = cv2.resize(mask_8bit, (image.shape[1], image.shape[0]))
        
        # --- NEW: Get the 32-element fingerprint ---
        fingerprint = get_color_fingerprint(image, mask_resized)
        
        # --- Use the DNN model to predict from the fingerprint ---
        color_name = color_model.get_color_name(fingerprint)
        
        x1, y1, x2, y2 = boxes[i][:4] 
        x_center = (x1 + x2) / 2
        
        if color_name:
            detected_bands.append((x_center, color_name))
            # <--- MODIFIED: Added confidence to the printout for clarity ---
            print(f"   - Band at x={x_center:.0f}: Color={color_name} (Conf: {confidence:.2f})")
        else:
            print(f"   - Band at x={x_center:.0f}: Could not determine color. (Conf: {confidence:.2f})")

    print("--- 5. Decoding...")
    resistance_value = decode_resistor(detected_bands)
    
    print(f"\n======================================")
    print(f" FINAL RESULT: {resistance_value}")
    print(f"======================================")
    
    cv2.putText(annotated_image, resistance_value, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    output_path = "resistor_result.jpg"
    cv2.imwrite(output_path, annotated_image)
    print(f"Result image saved to: {output_path}")

    cv2.imshow("Resistor Detection Result", annotated_image)
    print("Press any key to close the image window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ==============================================================================
# 4. RUN THE SCRIPT
# ==============================================================================

if __name__ == "__main__":
    
    print(">>> SCRIPT EXECUTION STARTED <<<")
    
    MODEL_PATH = r'D:/Hackathon/Band_Detection_V4/ResistorProject/run_heavy_aug/weights/best.pt' 
    IMAGE_PATH = r"D:\Hackathon\Band_Detection_V3\Real_Images\11.jpg"
    # --- Paths for the new Histogram DNN model ---
    DNN_MODEL_PATH = "color_classifier_histogram.keras"
    ASSETS_PATH = "color_classifier_histogram_assets.joblib"
    
    color_model = ColorClassifierDNN(
        model_path=DNN_MODEL_PATH,
        assets_path=ASSETS_PATH
    )

    if color_model.model is not None:
        process_image(MODEL_PATH, IMAGE_PATH, color_model)
    else:
        print("\nCould not run processing. Please create the color classifier model first.")