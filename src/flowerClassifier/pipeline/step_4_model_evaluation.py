from flowerClassifier.config.configuration import ConfigManager
from flowerClassifier.components.evaluate_model import EvaluateModel
from flowerClassifier import logger

STAGE_NAME = "Evaluate Trained Model"

class EvaluateModelPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigManager()
        model_evaluation_config = config.get_model_evaluation_config()
        evaluate_model = EvaluateModel(config=model_evaluation_config)
        evaluate_model.evaluate_model()
        evaluate_model.log_to_mlflow()
        
if __name__ == '__main__':
    try:
        logger.info(f"############# Stage: {STAGE_NAME} started #############")
        obj = EvaluateModelPipeline()
        obj.main()
        logger.info(f"############# Stage: {STAGE_NAME} completed #############")
        
    except Exception as e:
        logger.exception(e)
        raise(e)