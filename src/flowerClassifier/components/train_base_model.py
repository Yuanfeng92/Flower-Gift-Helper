from pathlib import Path
import tensorflow as tf
from flowerClassifier import logger
from flowerClassifier.utils.common import save_json
from flowerClassifier.entity.entity_class import (ModelTrainingConfig)


class TrainBaseModel():
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
    
    def save_mapping(self):
        save_json(path=Path("constant/class_mapping.json"), data=self.train_generator.class_indices)
    
    def train_valid_generator(self):
        image_data_generator_kwargs=dict(
            rescale=1./255,
            validation_split=self.config.params_validation_split
            )
            
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            **image_data_generator_kwargs
        )
        
        validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            **image_data_generator_kwargs
        )
        
        # configure data flow
        data_flow_kwargs = dict(
            target_size=self.config.params_input_shape[:2],
            batch_size=self.config.params_batch_size,
            class_mode='categorical',
            interpolation="bilinear"
        )
                
        self.train_generator = train_datagen.flow_from_directory(
            directory = self.config.data_dir,
            subset='training',
            shuffle=True,
            ** data_flow_kwargs
        )
        
        self.save_mapping()
        
        self.validation_generator = validation_datagen.flow_from_directory(
            directory = self.config.data_dir,
            subset='validation',
            shuffle=False,
            ** data_flow_kwargs
        )
        
    def train_model(self):
        self.model = tf.keras.models.load_model(self.config.updated_model_path)

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        if self.config.params_is_early_stopping:
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                              patience=5, 
                                                              restore_best_weights=True)
                
            self.model.fit(self.train_generator,
                           validation_data=self.validation_generator,
                           epochs=self.config.params_epochs,
                           callbacks=[early_stopping])
        else:
            self.model.fit(self.train_generator, 
                           validation_data=self.validation_generator,
                           epochs=self.config.params_epochs)
            
        self.save_model(self.config.trained_model_path, self.model)
        logger.info(f"Trained model successfully saved to folder {self.config.trained_model_path}")