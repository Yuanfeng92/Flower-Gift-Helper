from flowerClassifier.config.configuration import ConfigManager
from flowerClassifier.components.data_ingestion import DataIngestion
from flowerClassifier import logger

STAGE_NAME = "Data Ingestion"

class DataIngestionPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()
        data_ingestion.split_data()


if __name__ == '__main__':
    try:
        logger.info(f"############# Stage: {STAGE_NAME} started #############")
        obj = DataIngestionPipeline()
        obj.main()
        logger.info(f"############# Stage: {STAGE_NAME} completed #############")
        
    except Exception as e:
        logger.exception(e)
        raise(e)