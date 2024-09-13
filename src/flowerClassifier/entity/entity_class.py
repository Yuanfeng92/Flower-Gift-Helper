from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    folder_dir: Path
    source_URL: str
    data_file_path: Path
    unzip_folder_dir: Path
    data_dir: Path
    train_dir: Path
    test_dir: Path
    params_test_split: float

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    folder_dir: Path
    base_model_path: Path
    updated_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_num_classes: int
    
@dataclass(frozen=True)
class ModelTrainingConfig:
    data_dir: Path
    updated_model_path: Path
    trained_model_path: Path
    params_validation_split: float
    params_input_shape: list
    params_batch_size: int
    params_is_early_stopping: bool
    params_epochs: int
    params_learning_rate: float

@dataclass(frozen=True)
class ModelEvaluationConfig:
    trained_model_path: Path
    test_dir: Path
    params_input_shape: list
    params_batch_size: int
    params_all: dict