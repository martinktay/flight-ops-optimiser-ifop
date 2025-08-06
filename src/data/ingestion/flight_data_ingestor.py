"""
Flight data ingestion module for collecting flight operations data.

This module handles the ingestion of flight data from various sources including
databases, APIs, and file systems.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import requests
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.utils.config import Config
from src.utils.logging import get_logger

logger = get_logger(__name__)


class FlightDataIngestor:
    """
    A class to handle the ingestion of flight operations data from various sources.

    This class provides methods to ingest flight data from databases, APIs,
    and file systems with proper error handling and data validation.
    """

    def __init__(self, config: Config) -> None:
        """
        Initialise the FlightDataIngestor with configuration.

        Args:
            config: Configuration object containing data source settings
        """
        self.config = config
        self.engine: Optional[Engine] = None

    def connect_to_database(self, connection_string: str) -> bool:
        """
        Establish connection to the database.

        Args:
            connection_string: Database connection string

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.engine = create_engine(connection_string)
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Successfully connected to database")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False

    def ingest_from_database(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Ingest data from database using SQL query.

        Args:
            query: SQL query to execute
            params: Optional parameters for the query

        Returns:
            Optional[pd.DataFrame]: Ingested data or None if failed
        """
        if not self.engine:
            logger.error("No database connection established")
            return None

        try:
            df = pd.read_sql(query, self.engine, params=params)
            logger.info(
                f"Successfully ingested {len(df)} records from database")
            return df
        except Exception as e:
            logger.error(f"Failed to ingest data from database: {e}")
            return None

    def ingest_from_api(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Ingest data from REST API.

        Args:
            url: API endpoint URL
            headers: Optional request headers
            params: Optional query parameters

        Returns:
            Optional[pd.DataFrame]: Ingested data or None if failed
        """
        try:
            response = requests.get(
                url, headers=headers, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            df = pd.DataFrame(data)
            logger.info(f"Successfully ingested {len(df)} records from API")
            return df
        except Exception as e:
            logger.error(f"Failed to ingest data from API: {e}")
            return None

    def ingest_from_file(
        self,
        file_path: Union[str, Path],
        file_type: str = "csv"
    ) -> Optional[pd.DataFrame]:
        """
        Ingest data from file system.

        Args:
            file_path: Path to the data file
            file_type: Type of file (csv, excel, json, parquet)

        Returns:
            Optional[pd.DataFrame]: Ingested data or None if failed
        """
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return None

            if file_type.lower() == "csv":
                df = pd.read_csv(file_path)
            elif file_type.lower() in ["excel", "xlsx", "xls"]:
                df = pd.read_excel(file_path)
            elif file_type.lower() == "json":
                df = pd.read_json(file_path)
            elif file_type.lower() == "parquet":
                df = pd.read_parquet(file_path)
            else:
                logger.error(f"Unsupported file type: {file_type}")
                return None

            logger.info(
                f"Successfully ingested {len(df)} records from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Failed to ingest data from file {file_path}: {e}")
            return None

    def ingest_flight_schedule(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Ingest flight schedule data for a specific date range.

        Args:
            start_date: Start date for data ingestion
            end_date: End date for data ingestion

        Returns:
            Optional[pd.DataFrame]: Flight schedule data or None if failed
        """
        query = """
        SELECT 
            flight_id,
            origin_airport,
            destination_airport,
            scheduled_departure,
            scheduled_arrival,
            aircraft_type,
            airline_code
        FROM flight_schedule 
        WHERE scheduled_departure BETWEEN :start_date AND :end_date
        """

        params = {
            "start_date": start_date,
            "end_date": end_date
        }

        return self.ingest_from_database(query, params)

    def ingest_weather_data(
        self,
        airport_codes: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Ingest weather data for specific airports and date range.

        Args:
            airport_codes: List of airport codes
            start_date: Start date for weather data
            end_date: End date for weather data

        Returns:
            Optional[pd.DataFrame]: Weather data or None if failed
        """
        query = """
        SELECT 
            airport_code,
            timestamp,
            temperature,
            humidity,
            wind_speed,
            wind_direction,
            visibility,
            precipitation
        FROM weather_data 
        WHERE airport_code IN :airport_codes 
        AND timestamp BETWEEN :start_date AND :end_date
        """

        params = {
            "airport_codes": tuple(airport_codes),
            "start_date": start_date,
            "end_date": end_date
        }

        return self.ingest_from_database(query, params)

    def ingest_historical_delays(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Ingest historical flight delay data.

        Args:
            start_date: Start date for delay data
            end_date: End date for delay data

        Returns:
            Optional[pd.DataFrame]: Historical delay data or None if failed
        """
        query = """
        SELECT 
            flight_id,
            actual_departure,
            actual_arrival,
            departure_delay,
            arrival_delay,
            delay_reason,
            weather_conditions
        FROM flight_delays 
        WHERE actual_departure BETWEEN :start_date AND :end_date
        """

        params = {
            "start_date": start_date,
            "end_date": end_date
        }

        return self.ingest_from_database(query, params)

    def ingest_flights_dataset(
        self,
        file_path: str = "flights_sample_3m.csv",
        sample_size: Optional[int] = None,
        date_range: Optional[tuple] = None,
        airline_codes: Optional[List[str]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Ingest the flights dataset with optional filtering and sampling.

        This method loads the large flights dataset and optionally applies filters
        to make it more manageable for analysis and development.

        Args:
            file_path: Path to the flights CSV file
            sample_size: Number of rows to sample (None for full dataset)
            date_range: Tuple of (start_date, end_date) in YYYY-MM-DD format
            airline_codes: List of airline codes to filter by

        Returns:
            Optional[pd.DataFrame]: Filtered flights data or None if failed
        """
        try:
            logger.info(f"Ingesting flights dataset from {file_path}")

            # Load the dataset
            if sample_size:
                logger.info(f"Loading sample of {sample_size} rows")
                df = pd.read_csv(file_path, nrows=sample_size)
            else:
                logger.info("Loading full dataset")
                df = pd.read_csv(file_path)

            logger.info(f"Loaded {len(df)} rows from flights dataset")

            # Apply date filter if specified
            if date_range:
                start_date, end_date = date_range
                logger.info(
                    f"Filtering by date range: {start_date} to {end_date}")
                df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
                mask = (df['FL_DATE'] >= start_date) & (
                    df['FL_DATE'] <= end_date)
                df = df[mask].copy()
                logger.info(f"After date filter: {len(df)} rows")

            # Apply airline filter if specified
            if airline_codes:
                logger.info(f"Filtering by airlines: {airline_codes}")
                mask = df['AIRLINE_CODE'].isin(airline_codes)
                df = df[mask].copy()
                logger.info(f"After airline filter: {len(df)} rows")

            # Basic data validation
            required_columns = [
                'FL_DATE', 'AIRLINE_CODE', 'FL_NUMBER', 'ORIGIN', 'DEST',
                'DEP_DELAY', 'ARR_DELAY', 'DISTANCE'
            ]

            missing_columns = [
                col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return None

            logger.info(
                f"Successfully ingested flights dataset: {len(df)} rows, {len(df.columns)} columns")
            return df

        except Exception as e:
            logger.error(f"Failed to ingest flights dataset: {e}")
            return None
