#!/usr/bin/env python3
"""
Test script for the dataset filtering functionality.

This script demonstrates how to use the DatasetFilter class to create
manageable samples of the flights dataset for development and testing.
"""

import logging
from data.preprocessing.dataset_filter import DatasetFilter
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    """Main function to test dataset filtering."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("🚀 Testing Dataset Filter for Flight Operations Optimiser")
    print("=" * 60)

    # Create filter instance
    filter_tool = DatasetFilter()

    try:
        # Test 1: Quick sample for development (50K rows)
        print("\n📊 TEST 1: Development Sample (50,000 rows)")
        print("-" * 40)

        df_dev = filter_tool.generate_filtered_dataset(
            output_file="data/flights_dev_sample.csv",
            sample_size=50000,
            include_optional_columns=False
        )

        if df_dev is not None:
            stats_dev = filter_tool.generate_summary_statistics(df_dev)
            filter_tool.print_summary_report(stats_dev)
        else:
            print("❌ Failed to generate development sample")
            return

        # Test 2: Major airlines only (100K rows)
        print("\n📊 TEST 2: Major Airlines Sample (100,000 rows)")
        print("-" * 40)

        df_airlines = filter_tool.generate_filtered_dataset(
            output_file="data/flights_major_airlines.csv",
            # American, Delta, United, Southwest, Alaska
            airline_codes=['AA', 'DL', 'UA', 'WN', 'AS'],
            sample_size=100000,
            include_optional_columns=False
        )

        if df_airlines is not None:
            stats_airlines = filter_tool.generate_summary_statistics(
                df_airlines)
            filter_tool.print_summary_report(stats_airlines)
        else:
            print("❌ Failed to generate airlines sample")

        # Test 3: Recent data (last 3 months of available data)
        print("\n📊 TEST 3: Recent Data Sample (Last 3 months of available data)")
        print("-" * 40)

        df_recent = filter_tool.generate_filtered_dataset(
            output_file="data/flights_recent_2023.csv",
            date_range=("2023-06-01", "2023-08-31"),
            sample_size=75000,
            include_optional_columns=False
        )

        if df_recent is not None:
            stats_recent = filter_tool.generate_summary_statistics(df_recent)
            filter_tool.print_summary_report(stats_recent)
        else:
            print("❌ Failed to generate recent data sample")

        # Test 4: Full featured sample (with optional columns)
        print("\n📊 TEST 4: Full Featured Sample (25,000 rows with all columns)")
        print("-" * 40)

        df_full = filter_tool.generate_filtered_dataset(
            output_file="data/flights_full_featured.csv",
            sample_size=25000,
            include_optional_columns=True
        )

        if df_full is not None:
            stats_full = filter_tool.generate_summary_statistics(df_full)
            filter_tool.print_summary_report(stats_full)
        else:
            print("❌ Failed to generate full featured sample")

        print("\n✅ Dataset filtering tests completed successfully!")
        print("\n📁 Generated files:")
        print("   - data/flights_dev_sample.csv (50K rows, essential columns)")
        print("   - data/flights_major_airlines.csv (100K rows, major airlines)")
        print("   - data/flights_recent_2023.csv (75K rows, recent data)")
        print("   - data/flights_full_featured.csv (25K rows, all columns)")

        print("\n💡 Usage tips:")
        print("   - Use flights_dev_sample.csv for quick development and testing")
        print("   - Use flights_major_airlines.csv for airline-specific analysis")
        print("   - Use flights_recent_2023.csv for recent trends analysis")
        print("   - Use flights_full_featured.csv for detailed analysis with all features")

    except Exception as e:
        print(f"❌ Error during dataset filtering: {e}")
        logging.error(f"Dataset filtering failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    exit_code = main()
    sys.exit(exit_code)
