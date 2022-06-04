from tensorflow import keras
from data_extraction import class_names
import numpy as np


model = keras.models.load_model(f'models/classifier_VGG16.h5')

def get_most_probable_class(class_names,mask):
    
    assert len(class_names) == len(mask)
    
    return class_names[np.array(mask).argmax(axis=0)]
