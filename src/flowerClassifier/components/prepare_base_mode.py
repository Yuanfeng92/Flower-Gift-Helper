from pathlib import Path
import tensorflow as tf
from flowerClassifier import logger
from flowerClassifier.config.configuration import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
        
    def get_base_model(self):
        self.model = tf.keras.applications.MobileNetV2(
            include_top=self.config.params_include_top,
            weights=self.config.params_weights,
            input_shape=self.config.params_image_size,
            classes=self.config.params_num_classes,
            )
        
        self.save_model(path=self.config.base_model_path, model=self.model)
        logger.info(f"Base model successfully saved to folder {self.config.base_model_path}")
        
        
    @staticmethod
    def prepare_base_model(model, num_classes, freeze_all, freeze_till, learning_rate):
        
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:freeze_till]:
                layer.trainable = False
        
        x = tf.keras.layers.GlobalAveragePooling2D()(model.output)
        x = tf.keras.layers.Dense(128, activation = 'relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        prediction = tf.keras.layers.Dense(units=num_classes, activation='softmax')(x)
        
        custom_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )
        
        # custom_model.compile(
        #     optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        #     loss='categorical_crossentropy',
        #     metrics=['accuracy'])
        
        # custom_model.summary(show_trainable = True)
        
        return custom_model
    
    def update_base_model(self):
        self.updated_model = self.prepare_base_model(
            model = self.model,
            num_classes = self.config.params_num_classes,
            freeze_all = True,
            freeze_till = None,
            learning_rate = self.config.params_learning_rate
        )
        
        self.save_model(path = self.config.updated_model_path, model = self.updated_model)
        logger.info(f"Updated model successfully saved to folder {self.config.updated_model_path}")