import tensorflow as tf
import mlflow
from pathlib import Path
from flowerClassifier import logger
from flowerClassifier.utils.common import save_json
from flowerClassifier.entity.entity_class import ModelEvaluationConfig 

class EvaluateModel():
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.model_signature = None
        
    def create_test_generator(self):
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

        # configure data flow
        data_flow_kwargs = dict(
            target_size=self.config.params_input_shape[:2],
            batch_size=self.config.params_batch_size,
            class_mode='categorical',
            interpolation="bilinear"
        )

        self.test_generator = test_datagen.flow_from_directory(
            directory = self.config.test_dir,
            ** data_flow_kwargs
        )
        
    def create_signature(self):
        
        # Extract a single image and its label
        X_batch, y_batch = next(self.test_generator)
        X_sample=X_batch[0:1]
        y_sample=y_batch[0:1]

        # Get model predictions for the sample image
        y_pred=self.trained_model.predict(X_sample)

        # Infer the model signature using the input (X_sample) and the output (y_pred)
        self.model_signature = mlflow.models.signature.infer_signature(X_sample, y_pred)
        logger.info(f"Model signature created")
        
    def save_score(self):
        self.scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("metrics.json"), data=self.scores)
        
    def evaluate_model(self):
        self.trained_model = tf.keras.models.load_model(self.config.trained_model_path)
        self.create_test_generator()
        self.score = self.trained_model.evaluate(self.test_generator)
        logger.info(f'The accuracy of the model is: {self.score[1]*100:.2f}%')
        logger.info(f'The loss of the model is: {self.score[0]:.4f}')
        self.save_score()
        
    def log_to_mlflow(self):
        
        # Log the model with the inferred signature using MLflow
        with mlflow.start_run():
            
            # Log model and metrics to MLflow
            mlflow.log_params(self.config.params_all)
            mlflow.log_metrics(self.scores)
            
            # if signature is not created, create a signature
            if self.model_signature == None:
                self.create_signature()
                
            # Log the model
            mlflow.keras.log_model(self.trained_model, "mobilenetv2_model", signature=self.model_signature)
            logger.info("Model logged")