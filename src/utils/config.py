"""
Configuration management utility for the flight operations optimisation project.

This module provides functionality for loading and managing configuration
settings from YAML files and environment variables.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from dotenv import load_dotenv

from src.utils.logging import get_logger

logger = get_logger(__name__)


class Config:
    """
    A configuration management class for loading and accessing project settings.

    This class provides functionality for loading configuration from YAML files,
    environment variables, and providing default values with proper validation.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """
        Initialise the Config class with configuration file path.

        Args:
            config_path: Path to the configuration YAML file
        """
        self.config_path = Path(config_path) if config_path else Path(
            "configs/config.yaml")
        self.config_data: Dict[str, Any] = {}

        # Load environment variables
        load_dotenv()

        # Load configuration
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from YAML file and environment variables."""
        try:
            # Load from YAML file if it exists
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as file:
                    self.config_data = yaml.safe_load(file) or {}
                logger.info(f"Configuration loaded from {self.config_path}")
            else:
                logger.warning(
                    f"Configuration file not found at {self.config_path}, using defaults")
                self.config_data = {}

            # Override with environment variables
            self._load_from_env()

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            self.config_data = {}

    def _load_from_env(self) -> None:
        """Load configuration values from environment variables."""
        env_mappings = {
            'DATABASE_URL': 'database.connection_string',
            'MLFLOW_TRACKING_URI': 'mlflow.tracking_uri',
            'GUROBI_LICENSE': 'optimisation.gurobi_license',
            'LOG_LEVEL': 'logging.level',
            'OUTPUT_DIR': 'paths.output_directory',
            'DATA_DIR': 'paths.data_directory',
            'MODEL_DIR': 'paths.model_directory'
        }

        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value:
                self._set_nested_value(config_path, env_value)
                logger.debug(f"Loaded {env_var} from environment")

    def _set_nested_value(self, path: str, value: Any) -> None:
        """
        Set a nested configuration value using dot notation.

        Args:
            path: Configuration path using dot notation (e.g., 'database.host')
            value: Value to set
        """
        keys = path.split('.')
        current = self.config_data

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Args:
            key: Configuration key using dot notation (e.g., 'database.host')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        current = self.config_data

        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default

        return current

    def get_database_config(self) -> Dict[str, Any]:
        """
        Get database configuration.

        Returns:
            Dict containing database configuration
        """
        return {
            'connection_string': self.get('database.connection_string', 'sqlite:///flight_data.db'),
            'host': self.get('database.host', 'localhost'),
            'port': self.get('database.port', 5432),
            'database': self.get('database.name', 'flight_ops'),
            'username': self.get('database.username', ''),
            'password': self.get('database.password', ''),
            'pool_size': self.get('database.pool_size', 10),
            'max_overflow': self.get('database.max_overflow', 20)
        }

    def get_mlflow_config(self) -> Dict[str, Any]:
        """
        Get MLflow configuration.

        Returns:
            Dict containing MLflow configuration
        """
        return {
            'tracking_uri': self.get('mlflow.tracking_uri', 'sqlite:///mlflow.db'),
            'experiment_name': self.get('mlflow.experiment_name', 'flight_delay_prediction'),
            'artifact_location': self.get('mlflow.artifact_location', './mlruns'),
            'registry_uri': self.get('mlflow.registry_uri', None)
        }

    def get_optimisation_config(self) -> Dict[str, Any]:
        """
        Get optimisation configuration.

        Returns:
            Dict containing optimisation configuration
        """
        return {
            'gurobi_license': self.get('optimisation.gurobi_license', ''),
            'time_limit': self.get('optimisation.time_limit', 300),
            'mip_gap': self.get('optimisation.mip_gap', 0.01),
            'max_crew_duty_hours': self.get('optimisation.max_crew_duty_hours', 12),
            'min_crew_rest_hours': self.get('optimisation.min_crew_rest_hours', 10),
            'max_aircraft_utilization': self.get('optimisation.max_aircraft_utilization', 0.85),
            'delay_penalty': self.get('optimisation.delay_penalty', 1000),
            'crew_cost_per_hour': self.get('optimisation.crew_cost_per_hour', 50),
            'aircraft_cost_per_hour': self.get('optimisation.aircraft_cost_per_hour', 200)
        }

    def get_model_config(self) -> Dict[str, Any]:
        """
        Get machine learning model configuration.

        Returns:
            Dict containing model configuration
        """
        return {
            'test_size': self.get('model.test_size', 0.2),
            'random_state': self.get('model.random_state', 42),
            'cv_folds': self.get('model.cv_folds', 5),
            'feature_selection': self.get('model.feature_selection', True),
            'hyperparameter_tuning': self.get('model.hyperparameter_tuning', True),
            'model_types': self.get('model.types', ['random_forest', 'gradient_boosting', 'linear_regression'])
        }

    def get_data_config(self) -> Dict[str, Any]:
        """
        Get data processing configuration.

        Returns:
            Dict containing data configuration
        """
        return {
            'data_directory': self.get('paths.data_directory', './data'),
            'output_directory': self.get('paths.output_directory', './output'),
            'model_directory': self.get('paths.model_directory', './models'),
            'reports_directory': self.get('paths.reports_directory', './reports'),
            'cache_directory': self.get('paths.cache_directory', './cache'),
            'file_formats': self.get('data.file_formats', ['csv', 'parquet', 'json']),
            'encoding': self.get('data.encoding', 'utf-8'),
            'chunk_size': self.get('data.chunk_size', 10000)
        }

    def get_logging_config(self) -> Dict[str, Any]:
        """
        Get logging configuration.

        Returns:
            Dict containing logging configuration
        """
        return {
            'level': self.get('logging.level', 'INFO'),
            'format': self.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            'file': self.get('logging.file', './logs/flight_ops.log'),
            'max_size': self.get('logging.max_size', 10 * 1024 * 1024),  # 10MB
            'backup_count': self.get('logging.backup_count', 5),
            'console_output': self.get('logging.console_output', True)
        }

    def get_dagster_config(self) -> Dict[str, Any]:
        """
        Get Dagster configuration.

        Returns:
            Dict containing Dagster configuration
        """
        return {
            'workspace_file': self.get('dagster.workspace_file', 'workspace.yaml'),
            'module_name': self.get('dagster.module_name', 'src.dagster_pipeline'),
            'pipeline_name': self.get('dagster.pipeline_name', 'flight_ops_pipeline'),
            # Daily at 6 AM
            'schedule_interval': self.get('dagster.schedule_interval', '0 6 * * *'),
            'retry_policy': self.get('dagster.retry_policy', {'max_retries': 3, 'delay': 60})
        }

    def validate_config(self) -> Dict[str, Any]:
        """
        Validate configuration and return validation results.

        Returns:
            Dict containing validation results
        """
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }

        # Check required configurations
        required_configs = [
            ('database.connection_string', 'Database connection string'),
            ('mlflow.tracking_uri', 'MLflow tracking URI'),
            ('paths.data_directory', 'Data directory path')
        ]

        for config_key, description in required_configs:
            if not self.get(config_key):
                validation_results['warnings'].append(
                    f"{description} not configured")

        # Check file paths
        data_dir = self.get('paths.data_directory')
        if data_dir and not Path(data_dir).exists():
            validation_results['warnings'].append(
                f"Data directory does not exist: {data_dir}")

        # Check optimisation settings
        if self.get('optimisation.max_crew_duty_hours', 0) > 16:
            validation_results['warnings'].append(
                "Crew duty hours exceed recommended maximum")

        if self.get('optimisation.max_aircraft_utilization', 0) > 1.0:
            validation_results['errors'].append(
                "Aircraft utilisation cannot exceed 100%")
            validation_results['valid'] = False

        logger.info(
            f"Configuration validation completed - Valid: {validation_results['valid']}")
        return validation_results

    def save_config(self, filepath: Optional[Union[str, Path]] = None) -> bool:
        """
        Save current configuration to YAML file.

        Args:
            filepath: Path to save configuration file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            save_path = Path(filepath) if filepath else self.config_path
            save_path.parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, 'w', encoding='utf-8') as file:
                yaml.dump(self.config_data, file,
                          default_flow_style=False, indent=2)

            logger.info(f"Configuration saved to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False

    def create_default_config(self) -> Dict[str, Any]:
        """
        Create a default configuration template.

        Returns:
            Dict containing default configuration
        """
        default_config = {
            'database': {
                'connection_string': 'postgresql://user:password@localhost:5432/flight_ops',
                'host': 'localhost',
                'port': 5432,
                'name': 'flight_ops',
                'username': 'flight_user',
                'password': '',
                'pool_size': 10,
                'max_overflow': 20
            },
            'mlflow': {
                'tracking_uri': 'sqlite:///mlflow.db',
                'experiment_name': 'flight_delay_prediction',
                'artifact_location': './mlruns',
                'registry_uri': None
            },
            'optimisation': {
                'gurobi_license': '',
                'time_limit': 300,
                'mip_gap': 0.01,
                'max_crew_duty_hours': 12,
                'min_crew_rest_hours': 10,
                'max_aircraft_utilization': 0.85,
                'delay_penalty': 1000,
                'crew_cost_per_hour': 50,
                'aircraft_cost_per_hour': 200
            },
            'model': {
                'test_size': 0.2,
                'random_state': 42,
                'cv_folds': 5,
                'feature_selection': True,
                'hyperparameter_tuning': True,
                'types': ['random_forest', 'gradient_boosting', 'linear_regression']
            },
            'paths': {
                'data_directory': './data',
                'output_directory': './output',
                'model_directory': './models',
                'reports_directory': './reports',
                'cache_directory': './cache'
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': './logs/flight_ops.log',
                'max_size': 10485760,  # 10MB
                'backup_count': 5,
                'console_output': True
            },
            'dagster': {
                'workspace_file': 'workspace.yaml',
                'module_name': 'src.dagster_pipeline',
                'pipeline_name': 'flight_ops_pipeline',
                'schedule_interval': '0 6 * * *',
                'retry_policy': {
                    'max_retries': 3,
                    'delay': 60
                }
            },
            'data': {
                'file_formats': ['csv', 'parquet', 'json'],
                'encoding': 'utf-8',
                'chunk_size': 10000
            }
        }

        return default_config
