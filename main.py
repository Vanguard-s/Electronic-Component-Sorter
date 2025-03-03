import silence_tensorflow.auto
import tensorflow as tf

import cv2
from datetime import datetime
import os
from IMG_CLASFF import gen_img_classifier
from TRANSISTOR_OCR import OCR
from CAPACITOR import capacitor_class

def capture_image(output_dir):
    # Generate folder name with today's date
    current_date = datetime.now().strftime("%Y-%m-%d")
    folder_path = os.path.join(output_dir, current_date)

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Open camera varies with which camera is used to capture images 0, 1,2 etc.....
    cap = cv2.VideoCapture(0)
    
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Display the frame
        cv2.imshow('Press c to capture, q to quit', frame)

        # Read keyboard input continuously
        keyboard_input = cv2.waitKey(1)

        # If 'c' is pressed, capture and save the image
        if keyboard_input == ord('c'):
            # Generate timestamp
            timestamp = datetime.now().strftime("%H-%M-%S")

            # Save the captured image with today's date and timestamp in the filename
            image_filename = f"image_{timestamp}.jpg"
            image_path = os.path.join(folder_path, image_filename)
            cv2.imwrite(image_path, frame)

            #print("Image captured and saved at:", image_path)
            break  # Exit the loop after capturing the image

        # If 'q' is pressed, exit
        elif keyboard_input == ord('q'):
            break

    # Release the camera
    cap.release()
    # Close all OpenCV windows
    cv2.destroyAllWindows()

    return image_path

def further_classifier(image_path,class_name):
        function_map = {

        1: capacitor_class,
        3: OCR,
        6: OCR,

        }

        if class_name in function_map:
        # Call the corresponding function
            return function_map[class_name](image_path)  # Pass the image path as an argument 
        else:
            print("Invalid index returned by image classifier")

def main():
    # Example usage:
    output_dir = r'C:\Users\krish\Documents\Coding\Python\Hackathon\pub\Captured_Images' #change path
    image_path = capture_image(output_dir)

    class_name=gen_img_classifier(image_path)
    
    result_string = f"{class_name},{image_path}"
    return result_string




if __name__ == "__main__":
    name = main()
    print(name)
    




