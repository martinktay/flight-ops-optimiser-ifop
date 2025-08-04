"""
Dataset filtering and sampling utilities for the Intelligent Flight Operations Optimiser (iFOP).

This module provides functions to filter the large flights dataset to a manageable size
while preserving all essential features for delay prediction and optimisation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DatasetFilter:
    """
    A class to filter and sample the flights dataset for manageable analysis.

    This class provides methods to:
    - Filter by date range
    - Filter by airline codes
    - Take random samples
    - Select essential columns only
    - Generate summary statistics
    """

    # Essential columns for delay prediction and optimisation
    ESSENTIAL_COLUMNS = [
        'FL_DATE',           # Date of flight
        'AIRLINE',           # Airline name
        'AIRLINE_CODE',      # Airline code
        'FL_NUMBER',         # Flight number
        'ORIGIN',            # Origin airport
        'DEST',              # Destination airport
        'CRS_DEP_TIME',      # Scheduled departure time
        'DEP_TIME',          # Actual departure time
        'DEP_DELAY',         # Departure delay
        'CRS_ARR_TIME',      # Scheduled arrival time
        'ARR_TIME',          # Actual arrival time
        'ARR_DELAY',         # Arrival delay (target variable)
        'CANCELLED',         # Cancellation flag
        'CANCELLATION_CODE',  # Cancellation reason
        'DIVERTED',          # Diversion flag
        'DISTANCE',          # Flight distance
        'DELAY_DUE_CARRIER',  # Carrier delay
        'DELAY_DUE_WEATHER',  # Weather delay
        'DELAY_DUE_NAS',     # National Air System delay
        'DELAY_DUE_SECURITY',  # Security delay
        'DELAY_DUE_LATE_AIRCRAFT'  # Late aircraft delay
    ]

    # Optional columns for advanced analysis
    OPTIONAL_COLUMNS = [
        'TAXI_OUT',          # Taxi out time
        'WHEELS_OFF',        # Wheels off time
        'WHEELS_ON',         # Wheels on time
        'TAXI_IN',           # Taxi in time
        'AIR_TIME',          # Air time
        'ELAPSED_TIME',      # Elapsed time
        'CRS_ELAPSED_TIME'   # Scheduled elapsed time
    ]

    def __init__(self, input_file: str = "flights_sample_3m.csv"):
        """
        Initialise the dataset filter.

        Args:
            input_file: Path to the input CSV file
        """
        self.input_file = Path(input_file)
        self.filtered_data = None

    def load_data(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load the dataset with optional sampling for initial analysis.

        Args:
            sample_size: Number of rows to sample for initial analysis

        Returns:
            DataFrame with the loaded data
        """
        logger.info(f"Loading data from {self.input_file}")

        if sample_size:
            # Use pandas to sample rows efficiently
            df = pd.read_csv(self.input_file, nrows=sample_size)
            logger.info(f"Loaded sample of {len(df)} rows")
        else:
            df = pd.read_csv(self.input_file)
            logger.info(f"Loaded full dataset: {len(df)} rows")

        return df

    def filter_by_date_range(self, df: pd.DataFrame,
                             start_date: str,
                             end_date: str) -> pd.DataFrame:
        """
        Filter dataset by date range.

        Args:
            df: Input DataFrame
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            Filtered DataFrame
        """
        logger.info(f"Filtering by date range: {start_date} to {end_date}")

        # Convert FL_DATE to datetime
        df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])

        # Filter by date range
        mask = (df['FL_DATE'] >= start_date) & (df['FL_DATE'] <= end_date)
        filtered_df = df[mask].copy()

        logger.info(f"Date filter: {len(df)} -> {len(filtered_df)} rows")
        return filtered_df

    def filter_by_airlines(self, df: pd.DataFrame,
                           airline_codes: List[str]) -> pd.DataFrame:
        """
        Filter dataset by airline codes.

        Args:
            df: Input DataFrame
            airline_codes: List of airline codes to keep

        Returns:
            Filtered DataFrame
        """
        logger.info(f"Filtering by airlines: {airline_codes}")

        mask = df['AIRLINE_CODE'].isin(airline_codes)
        filtered_df = df[mask].copy()

        logger.info(f"Airline filter: {len(df)} -> {len(filtered_df)} rows")
        return filtered_df

    def take_random_sample(self, df: pd.DataFrame,
                           sample_size: int,
                           random_state: int = 42) -> pd.DataFrame:
        """
        Take a random sample from the dataset.

        Args:
            df: Input DataFrame
            sample_size: Number of rows to sample
            random_state: Random seed for reproducibility

        Returns:
            Sampled DataFrame
        """
        logger.info(f"Taking random sample of {sample_size} rows")

        if sample_size >= len(df):
            logger.warning(
                f"Sample size {sample_size} >= dataset size {len(df)}. Returning full dataset.")
            return df.copy()

        sampled_df = df.sample(n=sample_size, random_state=random_state).copy()
        logger.info(f"Random sample: {len(df)} -> {len(sampled_df)} rows")
        return sampled_df

    def select_columns(self, df: pd.DataFrame,
                       include_optional: bool = False) -> pd.DataFrame:
        """
        Select essential columns from the dataset.

        Args:
            df: Input DataFrame
            include_optional: Whether to include optional columns

        Returns:
            DataFrame with selected columns
        """
        columns_to_keep = self.ESSENTIAL_COLUMNS.copy()

        if include_optional:
            columns_to_keep.extend(self.OPTIONAL_COLUMNS)

        # Only keep columns that exist in the dataset
        available_columns = [
            col for col in columns_to_keep if col in df.columns]
        missing_columns = [
            col for col in columns_to_keep if col not in df.columns]

        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}")

        selected_df = df[available_columns].copy()
        logger.info(
            f"Selected {len(available_columns)} columns from {len(df.columns)} available")

        return selected_df

    def generate_filtered_dataset(self,
                                  output_file: str = "flights_filtered.csv",
                                  date_range: Optional[Tuple[str, str]] = None,
                                  airline_codes: Optional[List[str]] = None,
                                  sample_size: Optional[int] = None,
                                  include_optional_columns: bool = False,
                                  random_state: int = 42) -> pd.DataFrame:
        """
        Generate a filtered dataset according to specified criteria.

        Args:
            output_file: Output file path
            date_range: Tuple of (start_date, end_date) in YYYY-MM-DD format
            airline_codes: List of airline codes to keep
            sample_size: Number of rows to sample
            include_optional_columns: Whether to include optional columns
            random_state: Random seed for reproducibility

        Returns:
            Filtered DataFrame
        """
        logger.info("Starting dataset filtering process")

        # Load data
        df = self.load_data()

        # Apply filters in sequence
        if date_range:
            start_date, end_date = date_range
            df = self.filter_by_date_range(df, start_date, end_date)

        if airline_codes:
            df = self.filter_by_airlines(df, airline_codes)

        if sample_size:
            df = self.take_random_sample(df, sample_size, random_state)

        # Select columns
        df = self.select_columns(df, include_optional_columns)

        # Save filtered dataset
        output_path = Path(output_file)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved filtered dataset to {output_path}")
        logger.info(
            f"Final dataset: {len(df)} rows, {len(df.columns)} columns")

        self.filtered_data = df
        return df

    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for the dataset.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with summary statistics
        """
        logger.info("Generating summary statistics")

        # Convert FL_DATE to datetime if it's not already
        if 'FL_DATE' in df.columns:
            if df['FL_DATE'].dtype == 'object':
                df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])

        # Basic statistics
        stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
        }

        # Handle date range if FL_DATE exists and has data
        if 'FL_DATE' in df.columns and len(df) > 0:
            min_date = df['FL_DATE'].min()
            max_date = df['FL_DATE'].max()

            if pd.notna(min_date) and pd.notna(max_date):
                stats['date_range'] = {
                    'start': min_date.strftime('%Y-%m-%d'),
                    'end': max_date.strftime('%Y-%m-%d')
                }
            else:
                stats['date_range'] = {
                    'start': 'N/A',
                    'end': 'N/A'
                }
        else:
            stats['date_range'] = {
                'start': 'N/A',
                'end': 'N/A'
            }

        # Add other statistics
        stats.update({
            'airlines': {
                'count': df['AIRLINE_CODE'].nunique(),
                'top_5': df['AIRLINE_CODE'].value_counts().head(5).to_dict()
            },
            'airports': {
                'origins': df['ORIGIN'].nunique(),
                'destinations': df['DEST'].nunique(),
                'top_origins': df['ORIGIN'].value_counts().head(5).to_dict(),
                'top_destinations': df['DEST'].value_counts().head(5).to_dict()
            },
            'delays': {
                'departure_delays': {
                    'mean': df['DEP_DELAY'].mean(),
                    'median': df['DEP_DELAY'].median(),
                    'std': df['DEP_DELAY'].std(),
                    'positive_delays': (df['DEP_DELAY'] > 0).sum(),
                    'delay_rate': (df['DEP_DELAY'] > 0).mean()
                },
                'arrival_delays': {
                    'mean': df['ARR_DELAY'].mean(),
                    'median': df['ARR_DELAY'].median(),
                    'std': df['ARR_DELAY'].std(),
                    'positive_delays': (df['ARR_DELAY'] > 0).sum(),
                    'delay_rate': (df['ARR_DELAY'] > 0).mean()
                }
            },
            'cancellations': {
                'total_cancelled': df['CANCELLED'].sum(),
                'cancellation_rate': df['CANCELLED'].mean(),
                'cancellation_reasons': df['CANCELLATION_CODE'].value_counts().to_dict()
            },
            'diversions': {
                'total_diverted': df['DIVERTED'].sum(),
                'diversion_rate': df['DIVERTED'].mean()
            },
            'distance': {
                'mean': df['DISTANCE'].mean(),
                'median': df['DISTANCE'].median(),
                'std': df['DISTANCE'].std(),
                'min': df['DISTANCE'].min(),
                'max': df['DISTANCE'].max()
            }
        })

        return stats

    def print_summary_report(self, stats: Dict[str, Any]) -> None:
        """
        Print a formatted summary report.

        Args:
            stats: Summary statistics dictionary
        """
        print("\n" + "="*60)
        print("FLIGHT OPERATIONS DATASET SUMMARY REPORT")
        print("="*60)

        print(f"\n📊 DATASET OVERVIEW:")
        print(f"   Total Rows: {stats['total_rows']:,}")
        print(f"   Total Columns: {stats['total_columns']}")
        print(
            f"   Date Range: {stats['date_range']['start']} to {stats['date_range']['end']}")

        print(f"\n✈️  AIRLINES:")
        print(f"   Number of Airlines: {stats['airlines']['count']}")
        print(f"   Top 5 Airlines:")
        for airline, count in stats['airlines']['top_5'].items():
            print(f"     {airline}: {count:,} flights")

        print(f"\n🏢 AIRPORTS:")
        print(f"   Origin Airports: {stats['airports']['origins']}")
        print(f"   Destination Airports: {stats['airports']['destinations']}")
        print(f"   Top 5 Origin Airports:")
        for airport, count in stats['airports']['top_origins'].items():
            print(f"     {airport}: {count:,} flights")

        print(f"\n⏰ DELAY ANALYSIS:")
        dep_delays = stats['delays']['departure_delays']
        arr_delays = stats['delays']['arrival_delays']
        print(f"   Departure Delays:")
        print(f"     Mean: {dep_delays['mean']:.1f} minutes")
        print(f"     Median: {dep_delays['median']:.1f} minutes")
        print(f"     Delay Rate: {dep_delays['delay_rate']:.1%}")
        print(f"   Arrival Delays:")
        print(f"     Mean: {arr_delays['mean']:.1f} minutes")
        print(f"     Median: {arr_delays['median']:.1f} minutes")
        print(f"     Delay Rate: {arr_delays['delay_rate']:.1%}")

        print(f"\n❌ CANCELLATIONS:")
        print(
            f"   Total Cancelled: {stats['cancellations']['total_cancelled']:,}")
        print(
            f"   Cancellation Rate: {stats['cancellations']['cancellation_rate']:.2%}")

        print(f"\n🔄 DIVERSIONS:")
        print(f"   Total Diverted: {stats['diversions']['total_diverted']:,}")
        print(
            f"   Diversion Rate: {stats['diversions']['diversion_rate']:.2%}")

        print(f"\n📏 DISTANCE:")
        print(f"   Mean Distance: {stats['distance']['mean']:.0f} miles")
        print(
            f"   Range: {stats['distance']['min']:.0f} - {stats['distance']['max']:.0f} miles")

        print("\n" + "="*60)


def main():
    """
    Main function to demonstrate dataset filtering.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Create filter instance
    filter_tool = DatasetFilter()

    # Example 1: Filter by date range (1 month)
    print("\n🔍 EXAMPLE 1: Filter by date range (January 2023)")
    df_month = filter_tool.generate_filtered_dataset(
        output_file="flights_jan_2023.csv",
        date_range=("2023-01-01", "2023-01-31"),
        include_optional_columns=False
    )

    stats_month = filter_tool.generate_summary_statistics(df_month)
    filter_tool.print_summary_report(stats_month)

    # Example 2: Filter by airline and take sample
    print("\n🔍 EXAMPLE 2: Filter by major airlines and take sample")
    df_airlines = filter_tool.generate_filtered_dataset(
        output_file="flights_major_airlines_sample.csv",
        # American, Delta, United, Southwest
        airline_codes=['AA', 'DL', 'UA', 'WN'],
        sample_size=50000,
        include_optional_columns=False
    )

    stats_airlines = filter_tool.generate_summary_statistics(df_airlines)
    filter_tool.print_summary_report(stats_airlines)

    # Example 3: Random sample for development
    print("\n🔍 EXAMPLE 3: Random sample for development")
    df_dev = filter_tool.generate_filtered_dataset(
        output_file="flights_dev_sample.csv",
        sample_size=100000,
        include_optional_columns=True
    )

    stats_dev = filter_tool.generate_summary_statistics(df_dev)
    filter_tool.print_summary_report(stats_dev)


if __name__ == "__main__":
    main()
