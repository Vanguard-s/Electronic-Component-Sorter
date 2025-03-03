import silence_tensorflow.auto
import tensorflow as tf

from keras.models import load_model  
from PIL import Image, ImageOps  
import numpy as np

def capacitor_class(image_path):
    #define paths to model files and image file
    modelpath = r'C:\Users\krish\Documents\Coding\Python\Hackathon\pub\CAPACITOR\capacitor_models\keras_model.h5' #change path to where the model is stored 
    labelpath= r'C:\Users\krish\Documents\Coding\Python\Hackathon\pub\CAPACITOR\capacitor_models\labels.txt'    # change path to where the labels for the model is stored


    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # loading the model and class names
    model = load_model(modelpath, compile=False) 
    class_names = open(labelpath, "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


    image = Image.open(image_path).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data,verbose = 0)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    print(class_name[2:])
    return class_name[2:]

