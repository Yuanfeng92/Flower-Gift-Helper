import os
import zipfile
import gdown
import shutil
from sklearn.model_selection import train_test_split

from flowerClassifier import logger
from flowerClassifier.config.configuration import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    
    def download_file(self)-> str:
        '''
        Fetch data from the url
        '''
        try: 
            dataset_url=self.config.source_URL
            zip_download_dir=self.config.data_file_path
            os.makedirs(self.config.folder_dir, exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix+file_id, zip_download_dir)
            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")
        except Exception as e:
            raise e
        
    
    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_folder_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.data_file_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
            
            
    def split_data(self):
        # Create train and validation directories if they don't exist
        os.makedirs(self.config.train_dir, exist_ok=True)
        os.makedirs(self.config.test_dir, exist_ok=True)

        # Get all class folders
        classes = os.listdir(self.config.data_dir)

        for cls in classes:
            # Create class folders in train and validation directories
            os.makedirs(os.path.join(self.config.train_dir, cls), exist_ok=True)
            os.makedirs(os.path.join(self.config.test_dir, cls), exist_ok=True)

            # Get all images in the class folder
            images = os.listdir(os.path.join(self.config.data_dir, cls))

            # Split the images into train and validation sets
            train_images, val_images = train_test_split(images, test_size=self.config.params_test_split, random_state=42)

            # Move train images
            for img in train_images:
                src = os.path.join(self.config.data_dir, cls, img)
                dst = os.path.join(self.config.train_dir, cls, img)
                shutil.copy(src, dst)

            # Move validation images
            for img in val_images:
                src = os.path.join(self.config.data_dir, cls, img)
                dst = os.path.join(self.config.test_dir, cls, img)
                shutil.copy(src, dst)
                
        logger.info(f"Data split successfully completed.")