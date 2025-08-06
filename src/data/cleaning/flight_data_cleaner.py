"""
Flight data cleaning module for processing and transforming flight operations data.

This module provides comprehensive data cleaning functionality including
handling missing values, outliers, data type conversions, and feature engineering.
"""

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.utils.logging import get_logger

logger = get_logger(__name__)


class FlightDataCleaner:
    """
    A class to handle cleaning and preprocessing of flight operations data.

    This class provides methods for handling missing values, outliers,
    data type conversions, and feature engineering specific to flight data.
    """

    def __init__(self) -> None:
        """Initialise the FlightDataCleaner."""
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.cleaning_stats: Dict[str, Any] = {}

    def clean_flight_schedule_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean flight schedule data.

        Args:
            df: Raw flight schedule DataFrame

        Returns:
            pd.DataFrame: Cleaned flight schedule data
        """
        logger.info("Starting flight schedule data cleaning")

        # Create a copy to avoid modifying original data
        df_clean = df.copy()

        # Handle missing values
        df_clean = self._handle_missing_values(df_clean)

        # Convert data types
        df_clean = self._convert_data_types(df_clean)

        # Remove duplicates
        df_clean = self._remove_duplicates(df_clean)

        # Validate flight data
        df_clean = self._validate_flight_data(df_clean)

        # Feature engineering
        df_clean = self._engineer_features(df_clean)

        logger.info(
            f"Flight schedule cleaning completed. Shape: {df_clean.shape}")
        return df_clean

    def clean_weather_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean weather data.

        Args:
            df: Raw weather DataFrame

        Returns:
            pd.DataFrame: Cleaned weather data
        """
        logger.info("Starting weather data cleaning")

        df_clean = df.copy()

        # Handle missing values
        df_clean = self._handle_weather_missing_values(df_clean)

        # Convert data types
        df_clean = self._convert_weather_data_types(df_clean)

        # Remove outliers
        df_clean = self._remove_weather_outliers(df_clean)

        # Feature engineering
        df_clean = self._engineer_weather_features(df_clean)

        logger.info(
            f"Weather data cleaning completed. Shape: {df_clean.shape}")
        return df_clean

    def clean_delay_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean flight delay data.

        Args:
            df: Raw delay DataFrame

        Returns:
            pd.DataFrame: Cleaned delay data
        """
        logger.info("Starting delay data cleaning")

        df_clean = df.copy()

        # Handle missing values
        df_clean = self._handle_delay_missing_values(df_clean)

        # Convert data types
        df_clean = self._convert_delay_data_types(df_clean)

        # Calculate derived delay metrics
        df_clean = self._calculate_delay_metrics(df_clean)

        # Remove invalid delay records
        df_clean = self._remove_invalid_delays(df_clean)

        logger.info(f"Delay data cleaning completed. Shape: {df_clean.shape}")
        return df_clean

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in flight schedule data.

        Args:
            df: DataFrame with missing values

        Returns:
            pd.DataFrame: DataFrame with handled missing values
        """
        initial_missing = df.isnull().sum().sum()

        # Drop rows with critical missing values
        critical_columns = ['flight_id',
                            'origin_airport', 'destination_airport']
        df = df.dropna(subset=critical_columns)

        # Fill missing aircraft_type with 'Unknown'
        df['aircraft_type'] = df['aircraft_type'].fillna('Unknown')

        # Fill missing airline_code with 'Unknown'
        df['airline_code'] = df['airline_code'].fillna('Unknown')

        final_missing = df.isnull().sum().sum()
        logger.info(
            f"Handled {initial_missing - final_missing} missing values")

        return df

    def _handle_weather_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in weather data.

        Args:
            df: Weather DataFrame with missing values

        Returns:
            pd.DataFrame: Weather DataFrame with handled missing values
        """
        initial_missing = df.isnull().sum().sum()

        # Forward fill for time series data
        df = df.sort_values(['airport_code', 'timestamp'])
        df = df.groupby('airport_code').fillna(method='ffill')

        # Backward fill for remaining missing values
        df = df.groupby('airport_code').fillna(method='bfill')

        # Fill remaining missing values with median
        numeric_columns = ['temperature',
                           'humidity', 'wind_speed', 'visibility']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        final_missing = df.isnull().sum().sum()
        logger.info(
            f"Handled {initial_missing - final_missing} missing weather values")

        return df

    def _handle_delay_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in delay data.

        Args:
            df: Delay DataFrame with missing values

        Returns:
            pd.DataFrame: Delay DataFrame with handled missing values
        """
        initial_missing = df.isnull().sum().sum()

        # Drop rows with missing flight_id
        df = df.dropna(subset=['flight_id'])

        # Fill missing delay reasons
        df['delay_reason'] = df['delay_reason'].fillna('Unknown')

        # Fill missing weather conditions
        df['weather_conditions'] = df['weather_conditions'].fillna('Unknown')

        final_missing = df.isnull().sum().sum()
        logger.info(
            f"Handled {initial_missing - final_missing} missing delay values")

        return df

    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert data types for flight schedule data.

        Args:
            df: DataFrame with incorrect data types

        Returns:
            pd.DataFrame: DataFrame with correct data types
        """
        # Convert datetime columns
        datetime_columns = ['scheduled_departure', 'scheduled_arrival']
        for col in datetime_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Convert string columns
        string_columns = ['flight_id', 'origin_airport',
                          'destination_airport', 'aircraft_type', 'airline_code']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)

        return df

    def _convert_weather_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert data types for weather data.

        Args:
            df: Weather DataFrame with incorrect data types

        Returns:
            pd.DataFrame: Weather DataFrame with correct data types
        """
        # Convert timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        # Convert numeric columns
        numeric_columns = ['temperature', 'humidity',
                           'wind_speed', 'wind_direction', 'visibility']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def _convert_delay_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert data types for delay data.

        Args:
            df: Delay DataFrame with incorrect data types

        Returns:
            pd.DataFrame: Delay DataFrame with correct data types
        """
        # Convert datetime columns
        datetime_columns = ['actual_departure', 'actual_arrival']
        for col in datetime_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Convert numeric columns
        numeric_columns = ['departure_delay', 'arrival_delay']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate records from flight schedule data.

        Args:
            df: DataFrame with potential duplicates

        Returns:
            pd.DataFrame: DataFrame without duplicates
        """
        initial_count = len(df)
        df = df.drop_duplicates()
        final_count = len(df)

        logger.info(f"Removed {initial_count - final_count} duplicate records")
        return df

    def _validate_flight_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate flight schedule data for logical consistency.

        Args:
            df: Flight schedule DataFrame

        Returns:
            pd.DataFrame: Validated flight schedule data
        """
        initial_count = len(df)

        # Remove flights where departure is after arrival
        if 'scheduled_departure' in df.columns and 'scheduled_arrival' in df.columns:
            df = df[df['scheduled_departure'] < df['scheduled_arrival']]

        # Remove flights with same origin and destination
        if 'origin_airport' in df.columns and 'destination_airport' in df.columns:
            df = df[df['origin_airport'] != df['destination_airport']]

        final_count = len(df)
        logger.info(
            f"Removed {initial_count - final_count} invalid flight records")

        return df

    def _remove_weather_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove outliers from weather data using IQR method.

        Args:
            df: Weather DataFrame

        Returns:
            pd.DataFrame: Weather DataFrame without outliers
        """
        initial_count = len(df)

        numeric_columns = ['temperature',
                           'humidity', 'wind_speed', 'visibility']

        for col in numeric_columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

        final_count = len(df)
        logger.info(f"Removed {initial_count - final_count} weather outliers")

        return df

    def _calculate_delay_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate additional delay metrics.

        Args:
            df: Delay DataFrame

        Returns:
            pd.DataFrame: Delay DataFrame with additional metrics
        """
        # Calculate total delay
        if 'departure_delay' in df.columns and 'arrival_delay' in df.columns:
            df['total_delay'] = df['departure_delay'] + df['arrival_delay']

        # Calculate delay categories
        if 'total_delay' in df.columns:
            df['delay_category'] = pd.cut(
                df['total_delay'],
                bins=[-np.inf, 0, 15, 60, 120, np.inf],
                labels=['Early', 'Minor', 'Moderate', 'Significant', 'Major']
            )

        # Calculate delay duration
        if 'actual_departure' in df.columns and 'actual_arrival' in df.columns:
            df['delay_duration'] = (
                df['actual_arrival'] - df['actual_departure']).dt.total_seconds() / 60

        return df

    def _remove_invalid_delays(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove invalid delay records.

        Args:
            df: Delay DataFrame

        Returns:
            pd.DataFrame: Delay DataFrame without invalid records
        """
        initial_count = len(df)

        # Remove records with negative delays (early flights)
        if 'departure_delay' in df.columns:
            df = df[df['departure_delay'] >= 0]

        if 'arrival_delay' in df.columns:
            df = df[df['arrival_delay'] >= 0]

        # Remove records with extremely large delays (likely data errors)
        if 'total_delay' in df.columns:
            df = df[df['total_delay'] <= 1440]  # Max 24 hours

        final_count = len(df)
        logger.info(
            f"Removed {initial_count - final_count} invalid delay records")

        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for flight schedule data.

        Args:
            df: Flight schedule DataFrame

        Returns:
            pd.DataFrame: Flight schedule DataFrame with engineered features
        """
        # Flight duration
        if 'scheduled_departure' in df.columns and 'scheduled_arrival' in df.columns:
            df['flight_duration'] = (
                df['scheduled_arrival'] - df['scheduled_departure']).dt.total_seconds() / 60

        # Day of week
        if 'scheduled_departure' in df.columns:
            df['day_of_week'] = df['scheduled_departure'].dt.day_name()
            df['hour_of_day'] = df['scheduled_departure'].dt.hour

        # Season
        if 'scheduled_departure' in df.columns:
            df['month'] = df['scheduled_departure'].dt.month
            df['season'] = pd.cut(
                df['month'],
                bins=[0, 3, 6, 9, 12],
                labels=['Winter', 'Spring', 'Summer', 'Autumn']
            )

        # Encode categorical variables
        categorical_columns = ['aircraft_type',
                               'airline_code', 'day_of_week', 'season']
        for col in categorical_columns:
            if col in df.columns:
                df = self._encode_categorical(df, col)

        return df

    def _engineer_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for weather data.

        Args:
            df: Weather DataFrame

        Returns:
            pd.DataFrame: Weather DataFrame with engineered features
        """
        # Weather severity
        if 'wind_speed' in df.columns:
            df['wind_severity'] = pd.cut(
                df['wind_speed'],
                bins=[0, 10, 25, 50, np.inf],
                labels=['Calm', 'Light', 'Moderate', 'Strong']
            )

        # Visibility categories
        if 'visibility' in df.columns:
            df['visibility_category'] = pd.cut(
                df['visibility'],
                bins=[0, 1000, 5000, 10000, np.inf],
                labels=['Poor', 'Moderate', 'Good', 'Excellent']
            )

        # Temperature categories
        if 'temperature' in df.columns:
            df['temperature_category'] = pd.cut(
                df['temperature'],
                bins=[-np.inf, 0, 10, 20, 30, np.inf],
                labels=['Freezing', 'Cold', 'Cool', 'Warm', 'Hot']
            )

        # Encode categorical variables
        categorical_columns = ['wind_severity',
                               'visibility_category', 'temperature_category']
        for col in categorical_columns:
            if col in df.columns:
                df = self._encode_categorical(df, col)

        return df

    def _encode_categorical(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Encode categorical variables using LabelEncoder.

        Args:
            df: DataFrame containing categorical column
            column: Name of categorical column to encode

        Returns:
            pd.DataFrame: DataFrame with encoded categorical column
        """
        if column not in self.label_encoders:
            self.label_encoders[column] = LabelEncoder()
            df[f'{column}_encoded'] = self.label_encoders[column].fit_transform(
                df[column].astype(str))
        else:
            # Handle new categories by adding them to existing encoder
            existing_categories = set(self.label_encoders[column].classes_)
            new_categories = set(df[column].unique())
            all_categories = list(existing_categories.union(new_categories))

            # Re-fit encoder with all categories
            self.label_encoders[column] = LabelEncoder()
            self.label_encoders[column].fit(all_categories)
            df[f'{column}_encoded'] = self.label_encoders[column].transform(
                df[column].astype(str))

        return df

    def get_cleaning_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cleaning process.

        Returns:
            Dict[str, Any]: Cleaning statistics
        """
        return self.cleaning_stats.copy()
