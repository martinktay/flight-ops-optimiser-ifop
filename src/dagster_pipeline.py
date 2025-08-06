"""
Dagster pipeline for flight operations data processing and optimisation.

This module defines the Dagster pipeline that orchestrates the entire
flight operations data processing workflow including ingestion, cleaning,
modelling, and optimisation.
"""

from datetime import datetime, timedelta
from typing import Dict, Any

import dagster as dag
from dagster import (
    AssetExecutionContext,
    AssetIn,
    AssetOut,
    asset,
    multi_asset,
    Config,
    MetadataValue,
    Output
)
import pandas as pd

from src.data.ingestion.flight_data_ingestor import FlightDataIngestor
from src.data.cleaning.flight_data_cleaner import FlightDataCleaner
from src.models.delay_prediction.delay_predictor import DelayPredictor
from src.models.optimisation.scheduler import FlightScheduler
from src.visualisation.delay_analyser import DelayAnalyser
from src.utils.config import Config as AppConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


class FlightOpsConfig(Config):
    """Configuration for the flight operations pipeline."""

    start_date: str = "2024-01-01"
    end_date: str = "2024-01-31"
    airport_codes: list[str] = ["LHR", "JFK", "CDG", "NRT", "SYD"]
    model_type: str = "random_forest"
    optimisation_time_limit: int = 300


@asset(
    description="Raw flight schedule data ingested from various sources",
    group_name="data_ingestion"
)
def raw_flight_schedule(context: AssetExecutionContext, config: FlightOpsConfig) -> pd.DataFrame:
    """
    Ingest raw flight schedule data from database or files.

    Args:
        context: Dagster execution context
        config: Pipeline configuration

    Returns:
        pd.DataFrame: Raw flight schedule data
    """
    logger.info("Starting flight schedule data ingestion")

    # Load configuration
    app_config = AppConfig()

    # Initialise data ingestor
    ingestor = FlightDataIngestor(app_config)

    # Parse dates
    start_date = datetime.strptime(config.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(config.end_date, "%Y-%m-%d")

    # Ingest flight schedule data
    flight_data = ingestor.ingest_flight_schedule(start_date, end_date)

    if flight_data is None or flight_data.empty:
        raise ValueError("No flight schedule data retrieved")

    # Add metadata
    context.add_output_metadata({
        "num_records": MetadataValue.int(len(flight_data)),
        "date_range": MetadataValue.text(f"{start_date.date()} to {end_date.date()}"),
        "columns": MetadataValue.json(list(flight_data.columns))
    })

    logger.info(f"Ingested {len(flight_data)} flight schedule records")
    return flight_data


@asset(
    description="Raw weather data for flight operations analysis",
    group_name="data_ingestion"
)
def raw_weather_data(context: AssetExecutionContext, config: FlightOpsConfig) -> pd.DataFrame:
    """
    Ingest weather data for specified airports and date range.

    Args:
        context: Dagster execution context
        config: Pipeline configuration

    Returns:
        pd.DataFrame: Raw weather data
    """
    logger.info("Starting weather data ingestion")

    # Load configuration
    app_config = AppConfig()

    # Initialise data ingestor
    ingestor = FlightDataIngestor(app_config)

    # Parse dates
    start_date = datetime.strptime(config.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(config.end_date, "%Y-%m-%d")

    # Ingest weather data
    weather_data = ingestor.ingest_weather_data(
        config.airport_codes, start_date, end_date
    )

    if weather_data is None or weather_data.empty:
        raise ValueError("No weather data retrieved")

    # Add metadata
    context.add_output_metadata({
        "num_records": MetadataValue.int(len(weather_data)),
        "airports": MetadataValue.json(config.airport_codes),
        "date_range": MetadataValue.text(f"{start_date.date()} to {end_date.date()}")
    })

    logger.info(f"Ingested {len(weather_data)} weather records")
    return weather_data


@asset(
    description="Raw historical delay data for model training",
    group_name="data_ingestion"
)
def raw_delay_data(context: AssetExecutionContext, config: FlightOpsConfig) -> pd.DataFrame:
    """
    Ingest historical flight delay data for model training.

    Args:
        context: Dagster execution context
        config: Pipeline configuration

    Returns:
        pd.DataFrame: Raw delay data
    """
    logger.info("Starting delay data ingestion")

    # Load configuration
    app_config = AppConfig()

    # Initialise data ingestor
    ingestor = FlightDataIngestor(app_config)

    # Use the flights dataset with optional filtering
    # For development, use a sample; for production, use full dataset
    sample_size = 100000  # Adjust based on available memory and processing time

    # Ingest flights dataset using generated sample (this contains all the delay information)
    delay_data = ingestor.ingest_flights_dataset(
        file_path="data/flights_dev_sample.csv",  # Use generated development sample
        sample_size=None,  # No need for additional sampling since file is already filtered
        date_range=None,  # Use full date range for now
        airline_codes=None  # Use all airlines for now
    )

    if delay_data is None or delay_data.empty:
        raise ValueError("No delay data retrieved from flights dataset")

    # Add metadata
    context.add_output_metadata({
        "num_records": MetadataValue.int(len(delay_data)),
        "date_range": MetadataValue.text(f"{delay_data['FL_DATE'].min()} to {delay_data['FL_DATE'].max()}"),
        "airlines": MetadataValue.int(delay_data['AIRLINE_CODE'].nunique()),
        "airports": MetadataValue.int(delay_data['ORIGIN'].nunique()),
        "delay_columns": MetadataValue.json([col for col in delay_data.columns if 'delay' in col.lower()])
    })

    logger.info(
        f"Ingested {len(delay_data)} delay records from flights dataset")
    return delay_data


@asset(
    description="Cleaned and processed flight schedule data",
    group_name="data_processing",
    ins={"raw_flight_schedule": AssetIn()}
)
def cleaned_flight_schedule(context: AssetExecutionContext, raw_flight_schedule: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and process flight schedule data.

    Args:
        context: Dagster execution context
        raw_flight_schedule: Raw flight schedule data

    Returns:
        pd.DataFrame: Cleaned flight schedule data
    """
    logger.info("Starting flight schedule data cleaning")

    # Initialise data cleaner
    cleaner = FlightDataCleaner()

    # Clean flight schedule data
    cleaned_data = cleaner.clean_flight_schedule_data(raw_flight_schedule)

    # Add metadata
    context.add_output_metadata({
        "original_records": MetadataValue.int(len(raw_flight_schedule)),
        "cleaned_records": MetadataValue.int(len(cleaned_data)),
        "removed_records": MetadataValue.int(len(raw_flight_schedule) - len(cleaned_data)),
        "cleaning_stats": MetadataValue.json(cleaner.get_cleaning_stats())
    })

    logger.info(f"Cleaned flight schedule data: {len(cleaned_data)} records")
    return cleaned_data


@asset(
    description="Cleaned and processed weather data",
    group_name="data_processing",
    ins={"raw_weather_data": AssetIn()}
)
def cleaned_weather_data(context: AssetExecutionContext, raw_weather_data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and process weather data.

    Args:
        context: Dagster execution context
        raw_weather_data: Raw weather data

    Returns:
        pd.DataFrame: Cleaned weather data
    """
    logger.info("Starting weather data cleaning")

    # Initialise data cleaner
    cleaner = FlightDataCleaner()

    # Clean weather data
    cleaned_data = cleaner.clean_weather_data(raw_weather_data)

    # Add metadata
    context.add_output_metadata({
        "original_records": MetadataValue.int(len(raw_weather_data)),
        "cleaned_records": MetadataValue.int(len(cleaned_data)),
        "removed_records": MetadataValue.int(len(raw_weather_data) - len(cleaned_data))
    })

    logger.info(f"Cleaned weather data: {len(cleaned_data)} records")
    return cleaned_data


@asset(
    description="Cleaned and processed delay data",
    group_name="data_processing",
    ins={"raw_delay_data": AssetIn()}
)
def cleaned_delay_data(context: AssetExecutionContext, raw_delay_data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and process delay data.

    Args:
        context: Dagster execution context
        raw_delay_data: Raw delay data

    Returns:
        pd.DataFrame: Cleaned delay data
    """
    logger.info("Starting delay data cleaning")

    # Initialise data cleaner
    cleaner = FlightDataCleaner()

    # Clean delay data
    cleaned_data = cleaner.clean_delay_data(raw_delay_data)

    # Add metadata
    context.add_output_metadata({
        "original_records": MetadataValue.int(len(raw_delay_data)),
        "cleaned_records": MetadataValue.int(len(cleaned_data)),
        "removed_records": MetadataValue.int(len(raw_delay_data) - len(cleaned_data))
    })

    logger.info(f"Cleaned delay data: {len(cleaned_data)} records")
    return cleaned_data


@multi_asset(
    description="Trained delay prediction models and performance metrics",
    group_name="modelling",
    ins={
        "cleaned_flight_schedule": AssetIn(),
        "cleaned_weather_data": AssetIn(),
        "cleaned_delay_data": AssetIn()
    },
    outs={
        "trained_models": AssetOut(),
        "model_performance": AssetOut(),
        "feature_importance": AssetOut()
    }
)
def delay_prediction_models(
    context: AssetExecutionContext,
    cleaned_flight_schedule: pd.DataFrame,
    cleaned_weather_data: pd.DataFrame,
    cleaned_delay_data: pd.DataFrame,
    config: FlightOpsConfig
) -> tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame]:
    """
    Train delay prediction models using cleaned data.

    Args:
        context: Dagster execution context
        cleaned_flight_schedule: Cleaned flight schedule data
        cleaned_weather_data: Cleaned weather data
        cleaned_delay_data: Cleaned delay data
        config: Pipeline configuration

    Returns:
        tuple: Trained models, performance metrics, and feature importance
    """
    logger.info("Starting delay prediction model training")

    # Merge data for training
    # This is a simplified merge - in practice, you'd need more sophisticated joining logic
    training_data = cleaned_flight_schedule.merge(
        cleaned_delay_data, on='flight_id', how='inner', suffixes=('', '_delay')
    )

    # Merge with weather data if available
    if not cleaned_weather_data.empty:
        # This is a simplified merge - you'd need proper airport and timestamp matching
        training_data = training_data.merge(
            cleaned_weather_data, left_on='origin_airport', right_on='airport_code', how='left'
        )

    if training_data.empty:
        raise ValueError("No training data available after merging")

    # Initialise delay predictor
    predictor = DelayPredictor()

    # Train models
    trained_models = predictor.train_models(training_data)

    # Get performance metrics
    model_performance = predictor.get_model_performance()

    # Get feature importance for the best model
    feature_importance = predictor.get_feature_importance(config.model_type)

    # Add metadata
    context.add_output_metadata({
        "training_records": MetadataValue.int(len(training_data)),
        "models_trained": MetadataValue.json(list(trained_models.keys())),
        "best_model": MetadataValue.text(config.model_type),
        "best_model_r2": MetadataValue.float(model_performance.get(config.model_type, {}).get('r2', 0.0))
    })

    logger.info(f"Trained {len(trained_models)} delay prediction models")

    return trained_models, model_performance, feature_importance


@asset(
    description="Flight delay predictions for current schedule",
    group_name="prediction",
    ins={
        "cleaned_flight_schedule": AssetIn(),
        "cleaned_weather_data": AssetIn(),
        "delay_prediction_models": AssetIn()
    }
)
def delay_predictions(
    context: AssetExecutionContext,
    cleaned_flight_schedule: pd.DataFrame,
    cleaned_weather_data: pd.DataFrame,
    delay_prediction_models: Dict[str, Any],
    config: FlightOpsConfig
) -> pd.DataFrame:
    """
    Generate delay predictions for current flight schedule.

    Args:
        context: Dagster execution context
        cleaned_flight_schedule: Cleaned flight schedule data
        cleaned_weather_data: Cleaned weather data
        delay_prediction_models: Trained delay prediction models
        config: Pipeline configuration

    Returns:
        pd.DataFrame: Flight schedule with delay predictions
    """
    logger.info("Starting delay predictions")

    # Prepare data for prediction
    prediction_data = cleaned_flight_schedule.copy()

    # Merge with weather data if available
    if not cleaned_weather_data.empty:
        prediction_data = prediction_data.merge(
            cleaned_weather_data, left_on='origin_airport', right_on='airport_code', how='left'
        )

    # Initialise delay predictor
    predictor = DelayPredictor()

    # Load trained models
    predictor.models = delay_prediction_models

    # Generate predictions
    predictions = predictor.predict_delays(prediction_data, config.model_type)

    # Add predictions to data
    prediction_data['predicted_delay'] = predictions

    # Add metadata
    context.add_output_metadata({
        "flights_predicted": MetadataValue.int(len(prediction_data)),
        "average_predicted_delay": MetadataValue.float(prediction_data['predicted_delay'].mean()),
        "max_predicted_delay": MetadataValue.float(prediction_data['predicted_delay'].max()),
        "model_used": MetadataValue.text(config.model_type)
    })

    logger.info(
        f"Generated delay predictions for {len(prediction_data)} flights")
    return prediction_data


@asset(
    description="Optimised crew and aircraft schedule",
    group_name="optimisation",
    ins={"delay_predictions": AssetIn()}
)
def optimised_schedule(
    context: AssetExecutionContext,
    delay_predictions: pd.DataFrame,
    config: FlightOpsConfig
) -> pd.DataFrame:
    """
    Generate optimised crew and aircraft schedule.

    Args:
        context: Dagster execution context
        delay_predictions: Flight schedule with delay predictions
        config: Pipeline configuration

    Returns:
        pd.DataFrame: Optimised schedule
    """
    logger.info("Starting schedule optimisation")

    # Create sample crew and aircraft data (in practice, this would come from database)
    crews_df = pd.DataFrame({
        'crew_id': [f'crew_{i}' for i in range(1, 11)],
        'crew_type': ['pilot', 'co_pilot', 'flight_attendant'] * 3 + ['pilot'],
        'max_hours': [12, 12, 10] * 3 + [12]
    })

    aircraft_df = pd.DataFrame({
        'aircraft_id': [f'ac_{i}' for i in range(1, 6)],
        'aircraft_type': ['A320', 'B737', 'A350', 'B787', 'A380'],
        'max_hours': [16, 14, 18, 16, 20]
    })

    # Prepare flight data for optimisation
    flights_df = delay_predictions[[
        'flight_id', 'scheduled_departure', 'scheduled_arrival', 'predicted_delay']].copy()
    flights_df['flight_duration'] = (
        flights_df['scheduled_arrival'] - flights_df['scheduled_departure']
    ).dt.total_seconds() / 60

    # Initialise scheduler
    scheduler = FlightScheduler({
        'time_limit': config.optimisation_time_limit,
        'max_crew_duty_hours': 12,
        'min_crew_rest_hours': 10,
        'max_aircraft_utilization': 0.85
    })

    # Create and solve optimisation model
    model = scheduler.create_optimisation_model(
        flights_df, crews_df, aircraft_df)
    results = scheduler.solve_model()

    if not results:
        raise ValueError("Optimisation failed to find a solution")

    # Generate schedule report
    schedule_report = scheduler.generate_schedule_report()

    # Add metadata
    context.add_output_metadata({
        "flights_scheduled": MetadataValue.int(len(schedule_report)),
        "total_crew_hours": MetadataValue.float(schedule_report['crew_duty_hours'].sum()),
        "total_aircraft_hours": MetadataValue.float(schedule_report['aircraft_utilisation_hours'].sum()),
        "objective_value": MetadataValue.float(results.get('objective_value', 0.0)),
        "optimisation_status": MetadataValue.text(str(results.get('status', 'Unknown')))
    })

    logger.info(
        f"Generated optimised schedule for {len(schedule_report)} flights")
    return schedule_report


@asset(
    description="Delay analysis visualisations and reports",
    group_name="visualisation",
    ins={"delay_predictions": AssetIn(), "optimised_schedule": AssetIn()}
)
def delay_analysis_reports(
    context: AssetExecutionContext,
    delay_predictions: pd.DataFrame,
    optimised_schedule: pd.DataFrame
) -> Dict[str, Any]:
    """
    Generate delay analysis visualisations and reports.

    Args:
        context: Dagster execution context
        delay_predictions: Flight schedule with delay predictions
        optimised_schedule: Optimised schedule

    Returns:
        Dict: Analysis results and report paths
    """
    logger.info("Starting delay analysis and visualisation")

    # Initialise delay analyser
    analyser = DelayAnalyser()

    # Generate summary dashboard
    dashboard_fig = analyser.create_delay_summary_dashboard(delay_predictions)

    # Generate weather analysis if weather data is available
    weather_fig = None
    if 'temperature' in delay_predictions.columns:
        weather_fig = analyser.create_weather_delay_analysis(delay_predictions)

    # Generate operational performance report
    performance_fig = analyser.create_operational_performance_report(
        delay_predictions)

    # Generate summary statistics
    summary_stats = analyser.generate_summary_statistics(delay_predictions)

    # Prepare report paths
    report_paths = {
        "dashboard": "reports/visualisations/delay_summary_dashboard.png",
        "weather_analysis": "reports/visualisations/weather_delay_analysis.png" if weather_fig else None,
        "performance_report": "reports/visualisations/operational_performance_report.png"
    }

    # Add metadata
    context.add_output_metadata({
        "total_flights_analysed": MetadataValue.int(summary_stats.get('total_flights', 0)),
        "on_time_percentage": MetadataValue.float(summary_stats.get('on_time_percentage', 0.0)),
        "average_delay": MetadataValue.float(summary_stats.get('average_delay', 0.0)),
        "reports_generated": MetadataValue.json(list(report_paths.keys()))
    })

    logger.info("Generated delay analysis reports and visualisations")

    return {
        "summary_statistics": summary_stats,
        "report_paths": report_paths,
        "figures": {
            "dashboard": dashboard_fig,
            "weather_analysis": weather_fig,
            "performance_report": performance_fig
        }
    }


# Define the main pipeline
@dag.asset_group
def flight_ops_pipeline():
    """Main flight operations pipeline."""
    return [
        raw_flight_schedule,
        raw_weather_data,
        raw_delay_data,
        cleaned_flight_schedule,
        cleaned_weather_data,
        cleaned_delay_data,
        delay_prediction_models,
        delay_predictions,
        optimised_schedule,
        delay_analysis_reports
    ]
