"""
Main application entry point for the Flight Operations Optimiser.

This module provides the main application interface for running the
flight operations optimisation system with various modes and configurations.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from src.utils.config import Config
from src.utils.logging import setup_logging, get_logger
from src.data.ingestion.flight_data_ingestor import FlightDataIngestor
from src.data.cleaning.flight_data_cleaner import FlightDataCleaner
from src.models.delay_prediction.delay_predictor import DelayPredictor
from src.models.optimisation.scheduler import FlightScheduler
from src.visualisation.delay_analyser import DelayAnalyser

logger = get_logger(__name__)


class FlightOpsApp:
    """
    Main application class for the Flight Operations Optimiser.

    This class provides a unified interface for running different components
    of the flight operations optimisation system.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialise the Flight Operations Optimiser application.

        Args:
            config_path: Path to configuration file
        """
        self.config = Config(config_path)
        self.setup_logging()

        # Initialise components
        self.ingestor = FlightDataIngestor(self.config)
        self.cleaner = FlightDataCleaner()
        self.predictor = DelayPredictor(self.config.get_model_config())
        self.scheduler = FlightScheduler(self.config.get_optimisation_config())
        self.analyser = DelayAnalyser()

        logger.info("Flight Operations Optimiser initialised successfully")

    def setup_logging(self) -> None:
        """Set up logging configuration."""
        logging_config = self.config.get_logging_config()
        setup_logging(
            level=logging_config['level'],
            log_file=logging_config['file'],
            console_output=logging_config['console_output']
        )

    def run_data_pipeline(self, start_date: str, end_date: str, airport_codes: list[str]) -> None:
        """
        Run the complete data pipeline including ingestion and cleaning.

        Args:
            start_date: Start date for data ingestion (YYYY-MM-DD)
            end_date: End date for data ingestion (YYYY-MM-DD)
            airport_codes: List of airport codes to process
        """
        logger.info("Starting data pipeline execution")

        try:
            # Ingest data
            logger.info("Ingesting flight schedule data")
            flight_data = self.ingestor.ingest_flight_schedule(
                start_date, end_date)

            logger.info("Ingesting weather data")
            weather_data = self.ingestor.ingest_weather_data(
                airport_codes, start_date, end_date)

            logger.info("Ingesting historical delay data")
            delay_data = self.ingestor.ingest_historical_delays(
                start_date, end_date)

            # Clean data
            logger.info("Cleaning flight schedule data")
            cleaned_flights = self.cleaner.clean_flight_schedule_data(
                flight_data)

            logger.info("Cleaning weather data")
            cleaned_weather = self.cleaner.clean_weather_data(weather_data)

            logger.info("Cleaning delay data")
            cleaned_delays = self.cleaner.clean_delay_data(delay_data)

            logger.info("Data pipeline completed successfully")

            return cleaned_flights, cleaned_weather, cleaned_delays

        except Exception as e:
            logger.error(f"Data pipeline failed: {e}")
            raise

    def run_model_training(self, training_data) -> None:
        """
        Run model training pipeline.

        Args:
            training_data: Training data DataFrame
        """
        logger.info("Starting model training")

        try:
            # Train models
            trained_models = self.predictor.train_models(training_data)

            # Get performance metrics
            performance = self.predictor.get_model_performance()

            # Get feature importance
            feature_importance = self.predictor.get_feature_importance()

            logger.info("Model training completed successfully")
            logger.info(f"Model performance: {performance}")

            return trained_models, performance, feature_importance

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise

    def run_optimisation(self, flights_df, crews_df, aircraft_df) -> None:
        """
        Run optimisation pipeline.

        Args:
            flights_df: Flight schedule DataFrame
            crews_df: Crew information DataFrame
            aircraft_df: Aircraft information DataFrame
        """
        logger.info("Starting optimisation pipeline")

        try:
            # Create optimisation model
            model = self.scheduler.create_optimisation_model(
                flights_df, crews_df, aircraft_df)

            # Solve optimisation problem
            results = self.scheduler.solve_model()

            # Generate schedule report
            schedule_report = self.scheduler.generate_schedule_report()

            logger.info("Optimisation completed successfully")
            logger.info(
                f"Objective value: {results.get('objective_value', 0)}")

            return results, schedule_report

        except Exception as e:
            logger.error(f"Optimisation failed: {e}")
            raise

    def run_visualisation(self, data_df) -> None:
        """
        Run visualisation pipeline.

        Args:
            data_df: Data DataFrame for visualisation
        """
        logger.info("Starting visualisation pipeline")

        try:
            # Generate delay analysis dashboard
            dashboard = self.analyser.create_delay_summary_dashboard(data_df)

            # Generate weather analysis if weather data is available
            if 'temperature' in data_df.columns:
                weather_analysis = self.analyser.create_weather_delay_analysis(
                    data_df)

            # Generate operational performance report
            performance_report = self.analyser.create_operational_performance_report(
                data_df)

            # Generate summary statistics
            summary_stats = self.analyser.generate_summary_statistics(data_df)

            logger.info("Visualisation pipeline completed successfully")
            logger.info(f"Summary statistics: {summary_stats}")

            return dashboard, summary_stats

        except Exception as e:
            logger.error(f"Visualisation failed: {e}")
            raise

    def run_full_pipeline(self, start_date: str, end_date: str, airport_codes: list[str]) -> None:
        """
        Run the complete flight operations optimisation pipeline.

        Args:
            start_date: Start date for data ingestion (YYYY-MM-DD)
            end_date: End date for data ingestion (YYYY-MM-DD)
            airport_codes: List of airport codes to process
        """
        logger.info("Starting full pipeline execution")

        try:
            # Step 1: Data Pipeline
            cleaned_flights, cleaned_weather, cleaned_delays = self.run_data_pipeline(
                start_date, end_date, airport_codes
            )

            # Step 2: Model Training
            trained_models, performance, feature_importance = self.run_model_training(
                cleaned_flights
            )

            # Step 3: Generate Predictions
            predictions = self.predictor.predict_delays(cleaned_flights)
            cleaned_flights['predicted_delay'] = predictions

            # Step 4: Optimisation (with sample data)
            # In practice, you would load real crew and aircraft data
            crews_df = self._create_sample_crew_data()
            aircraft_df = self._create_sample_aircraft_data()

            results, schedule_report = self.run_optimisation(
                cleaned_flights, crews_df, aircraft_df
            )

            # Step 5: Visualisation
            dashboard, summary_stats = self.run_visualisation(cleaned_flights)

            logger.info("Full pipeline completed successfully")

            return {
                'cleaned_data': (cleaned_flights, cleaned_weather, cleaned_delays),
                'models': (trained_models, performance, feature_importance),
                'optimisation': (results, schedule_report),
                'visualisation': (dashboard, summary_stats)
            }

        except Exception as e:
            logger.error(f"Full pipeline failed: {e}")
            raise

    def _create_sample_crew_data(self):
        """Create sample crew data for testing."""
        import pandas as pd

        return pd.DataFrame({
            'crew_id': [f'crew_{i}' for i in range(1, 11)],
            'crew_type': ['pilot', 'co_pilot', 'flight_attendant'] * 3 + ['pilot'],
            'max_hours': [12, 12, 10] * 3 + [12]
        })

    def _create_sample_aircraft_data(self):
        """Create sample aircraft data for testing."""
        import pandas as pd

        return pd.DataFrame({
            'aircraft_id': [f'ac_{i}' for i in range(1, 6)],
            'aircraft_type': ['A320', 'B737', 'A350', 'B787', 'A380'],
            'max_hours': [16, 14, 18, 16, 20]
        })


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="Flight Operations Optimiser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python -m src.main --mode full --start-date 2024-01-01 --end-date 2024-01-31 --airports LHR JFK CDG
  
  # Run data pipeline only
  python -m src.main --mode data --start-date 2024-01-01 --end-date 2024-01-31
  
  # Run model training only
  python -m src.main --mode training --data-file data/flights.csv
  
  # Run optimisation only
  python -m src.main --mode optimisation --flights-file data/flights.csv --crews-file data/crews.csv
        """
    )

    parser.add_argument(
        '--mode',
        choices=['full', 'data', 'training', 'optimisation', 'visualisation'],
        default='full',
        help='Pipeline mode to run'
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )

    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for data ingestion (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for data ingestion (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--airports',
        nargs='+',
        help='List of airport codes to process'
    )

    parser.add_argument(
        '--data-file',
        type=str,
        help='Path to data file for training'
    )

    parser.add_argument(
        '--flights-file',
        type=str,
        help='Path to flights data file'
    )

    parser.add_argument(
        '--crews-file',
        type=str,
        help='Path to crews data file'
    )

    parser.add_argument(
        '--aircraft-file',
        type=str,
        help='Path to aircraft data file'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output',
        help='Output directory for results'
    )

    args = parser.parse_args()

    try:
        # Initialise application
        app = FlightOpsApp(args.config)

        # Run based on mode
        if args.mode == 'full':
            if not all([args.start_date, args.end_date, args.airports]):
                logger.error(
                    "Full mode requires start-date, end-date, and airports")
                sys.exit(1)

            results = app.run_full_pipeline(
                args.start_date, args.end_date, args.airports)
            logger.info("Full pipeline completed successfully")

        elif args.mode == 'data':
            if not all([args.start_date, args.end_date, args.airports]):
                logger.error(
                    "Data mode requires start-date, end-date, and airports")
                sys.exit(1)

            app.run_data_pipeline(
                args.start_date, args.end_date, args.airports)
            logger.info("Data pipeline completed successfully")

        elif args.mode == 'training':
            if not args.data_file:
                logger.error("Training mode requires data-file")
                sys.exit(1)

            import pandas as pd
            data = pd.read_csv(args.data_file)
            app.run_model_training(data)
            logger.info("Model training completed successfully")

        elif args.mode == 'optimisation':
            if not all([args.flights_file, args.crews_file, args.aircraft_file]):
                logger.error(
                    "Optimisation mode requires flights-file, crews-file, and aircraft-file")
                sys.exit(1)

            import pandas as pd
            flights = pd.read_csv(args.flights_file)
            crews = pd.read_csv(args.crews_file)
            aircraft = pd.read_csv(args.aircraft_file)

            app.run_optimisation(flights, crews, aircraft)
            logger.info("Optimisation completed successfully")

        elif args.mode == 'visualisation':
            if not args.data_file:
                logger.error("Visualisation mode requires data-file")
                sys.exit(1)

            import pandas as pd
            data = pd.read_csv(args.data_file)
            app.run_visualisation(data)
            logger.info("Visualisation completed successfully")

        logger.info("Application completed successfully")

    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
