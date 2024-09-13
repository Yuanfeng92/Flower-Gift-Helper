from flowerClassifier.config.configuration import ConfigManager
from flowerClassifier.components.train_base_model import TrainBaseModel
from flowerClassifier import logger

STAGE_NAME = "Train Base Model"

class TrainBaseModelPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigManager()
        model_training_config = config.get_model_training_config()
        train_base_model = TrainBaseModel(config=model_training_config)
        train_base_model.train_valid_generator()
        train_base_model.train_model()


if __name__ == '__main__':
    try:
        logger.info(f"############# Stage: {STAGE_NAME} started #############")
        obj = TrainBaseModelPipeline()
        obj.main()
        logger.info(f"############# Stage: {STAGE_NAME} completed #############")
        
    except Exception as e:
        logger.exception(e)
        raise(e)