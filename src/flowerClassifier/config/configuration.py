from flowerClassifier import logger, CONFIG_FILE_PATH, PARAMS_FILE_PATH
from flowerClassifier.utils.common import read_yaml, create_directories
from flowerClassifier.entity.entity_class import (DataIngestionConfig, PrepareBaseModelConfig, ModelTrainingConfig, ModelEvaluationConfig)

class ConfigManager():
    def __init__(self, config_filepath = CONFIG_FILE_PATH, params_filepath = PARAMS_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        
        create_directories([self.config.artifacts_root_dir])
        
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        
        create_directories([config.folder_dir])
        
        data_ingestion_config = DataIngestionConfig(
            folder_dir=config.folder_dir,
            source_URL=config.source_URL,
            data_file_path=config.data_file_path,
            unzip_folder_dir=config.unzip_folder_dir,
            data_dir=config.data_dir,
            train_dir=config.train_dir,
            test_dir=config.test_dir,
            params_test_split=self.params.TEST_SPLIT,
        )
        
        return data_ingestion_config
        
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.folder_dir])
        
        prepare_base_model_config = PrepareBaseModelConfig(
            folder_dir=config.folder_dir,
            base_model_path=config.base_model_path,
            updated_model_path=config.updated_model_path,
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_num_classes=self.params.NUM_CLASSES
        )
        
        return prepare_base_model_config
    
    def get_model_training_config(self) -> ModelTrainingConfig:
        
        create_directories([self.config.prepare_base_model.folder_dir])
        
        model_training_config = ModelTrainingConfig(
                data_dir=self.config.data_ingestion.data_dir,
                updated_model_path=self.config.prepare_base_model.updated_model_path,
                trained_model_path=self.config.model_training.trained_model_path,
                params_validation_split=self.params.VALIDATION_SPLIT,
                params_input_shape=self.params.IMAGE_SIZE,
                params_batch_size=self.params.BATCH_SIZE,
                params_is_early_stopping=self.params.EARLY_STOPPING,
                params_epochs=self.params.EPOCHS,
                params_learning_rate=self.params.LEARNING_RATE
        )
        
        return model_training_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
                
        model_evaluation_config = ModelEvaluationConfig(
                trained_model_path=self.config.model_training.trained_model_path,
                test_dir=self.config.data_ingestion.data_dir,
                params_input_shape=self.params.IMAGE_SIZE,
                params_batch_size=self.params.BATCH_SIZE,
                params_all=self.params
        )
        
        return model_evaluation_config

