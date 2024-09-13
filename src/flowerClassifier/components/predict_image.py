import tensorflow as tf
from pathlib import Path
import numpy as np
from flowerClassifier.utils.common import load_json
from flowerClassifier.config.configuration import ConfigManager
from flowerClassifier.entity.entity_class import ModelEvaluationConfig 


class ImageClassifier():
    def __init__(self):
        
        config = ConfigManager()
        model_evaluation_config = config.get_model_evaluation_config()
        self.config = model_evaluation_config
        
        self.trained_model = tf.keras.models.load_model(self.config.trained_model_path)
        self.class_mapping = load_json(Path("constant/class_mapping.json"))
        self.class_mapping_reverse = {}
        for key, value in self.class_mapping.items():
            self.class_mapping_reverse[value] = key
            
        print()
        
    def predict(self, image):
        
        # converts an input image (typically in the form of a PIL image or other image-like format) into a NumPy array
        input_image = tf.keras.preprocessing.image.img_to_array(image)
        input_image = tf.image.resize(input_image, size=self.config.params_input_shape[:2])
        # Normalize by dividing by 255 to match the evaluation pipeline
        input_image = input_image / 255.0 
        input_image = np.expand_dims(input_image, axis = 0)
        
        model_pred =  self.trained_model.predict(input_image)
        prediction_index = np.argmax(model_pred, axis=1)[0]
        prediction_class = self.class_mapping_reverse[prediction_index]
        
        return prediction_class
    
    # def predict(self, image_path):
        
    #     input_image = tf.keras.preprocessing.image.load_img(image_path, target_size =self.config.params_input_shape[:2])
    #     input_image = tf.keras.preprocessing.image.img_to_array(input_image)
    #     input_image = np.expand_dims(input_image, axis = 0)
        
    #     # Normalize by dividing by 255 to match the evaluation pipeline
    #     input_image = input_image / 255.0
        
    #     model_pred =  self.trained_model.predict(input_image)
    #     prediction_index = np.argmax(model_pred, axis=1)[0]
    #     prediction_class = self.class_mapping_reverse[prediction_index]
        
    #     return prediction_class