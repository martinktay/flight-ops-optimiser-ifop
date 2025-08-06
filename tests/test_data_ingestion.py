"""
Unit tests for data ingestion functionality.

This module contains comprehensive tests for the FlightDataIngestor class
and related data ingestion functionality.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy.engine import Engine

from src.data.ingestion.flight_data_ingestor import FlightDataIngestor
from src.utils.config import Config


class TestFlightDataIngestor:
    """Test cases for FlightDataIngestor class."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return Config()

    @pytest.fixture
    def ingestor(self, config):
        """Create a FlightDataIngestor instance for testing."""
        return FlightDataIngestor(config)

    @pytest.fixture
    def sample_flight_data(self):
        """Create sample flight data for testing."""
        return pd.DataFrame({
            'flight_id': ['FL001', 'FL002', 'FL003'],
            'origin_airport': ['LHR', 'JFK', 'CDG'],
            'destination_airport': ['JFK', 'LHR', 'NRT'],
            'scheduled_departure': [
                datetime(2024, 1, 1, 10, 0),
                datetime(2024, 1, 1, 14, 0),
                datetime(2024, 1, 1, 18, 0)
            ],
            'scheduled_arrival': [
                datetime(2024, 1, 1, 13, 0),
                datetime(2024, 1, 1, 17, 0),
                datetime(2024, 1, 2, 8, 0)
            ],
            'aircraft_type': ['A320', 'B737', 'A350'],
            'airline_code': ['BA', 'AA', 'AF']
        })

    @pytest.fixture
    def sample_weather_data(self):
        """Create sample weather data for testing."""
        return pd.DataFrame({
            'airport_code': ['LHR', 'JFK', 'CDG'],
            'timestamp': [
                datetime(2024, 1, 1, 10, 0),
                datetime(2024, 1, 1, 14, 0),
                datetime(2024, 1, 1, 18, 0)
            ],
            'temperature': [15.5, 22.3, 18.7],
            'humidity': [65, 45, 70],
            'wind_speed': [12.5, 8.2, 15.1],
            'wind_direction': [180, 270, 90],
            'visibility': [10000, 15000, 8000],
            'precipitation': [0.0, 0.0, 2.5]
        })

    def test_initialisation(self, ingestor):
        """Test FlightDataIngestor initialisation."""
        assert ingestor.config is not None
        assert ingestor.engine is None

    @patch('src.data.ingestion.flight_data_ingestor.create_engine')
    def test_connect_to_database_success(self, mock_create_engine, ingestor):
        """Test successful database connection."""
        # Mock the engine and connection
        mock_engine = Mock(spec=Engine)
        mock_connection = Mock()
        mock_connection.execute.return_value = None
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine

        # Test connection
        result = ingestor.connect_to_database("sqlite:///test.db")

        assert result is True
        assert ingestor.engine == mock_engine
        mock_create_engine.assert_called_once_with("sqlite:///test.db")

    @patch('src.data.ingestion.flight_data_ingestor.create_engine')
    def test_connect_to_database_failure(self, mock_create_engine, ingestor):
        """Test database connection failure."""
        # Mock connection failure
        mock_create_engine.side_effect = Exception("Connection failed")

        # Test connection
        result = ingestor.connect_to_database("invalid://connection")

        assert result is False
        assert ingestor.engine is None

    @patch('pandas.read_sql')
    def test_ingest_from_database_success(self, mock_read_sql, ingestor, sample_flight_data):
        """Test successful data ingestion from database."""
        # Mock the engine and pandas.read_sql
        ingestor.engine = Mock(spec=Engine)
        mock_read_sql.return_value = sample_flight_data

        # Test ingestion
        result = ingestor.ingest_from_database("SELECT * FROM flights")

        assert result is not None
        assert len(result) == 3
        assert list(result.columns) == list(sample_flight_data.columns)
        mock_read_sql.assert_called_once()

    def test_ingest_from_database_no_connection(self, ingestor):
        """Test database ingestion without connection."""
        result = ingestor.ingest_from_database("SELECT * FROM flights")
        assert result is None

    @patch('pandas.read_sql')
    def test_ingest_from_database_failure(self, mock_read_sql, ingestor):
        """Test database ingestion failure."""
        # Mock the engine and pandas.read_sql failure
        ingestor.engine = Mock(spec=Engine)
        mock_read_sql.side_effect = Exception("Query failed")

        # Test ingestion
        result = ingestor.ingest_from_database("SELECT * FROM flights")

        assert result is None

    @patch('requests.get')
    def test_ingest_from_api_success(self, mock_get, ingestor, sample_flight_data):
        """Test successful data ingestion from API."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.json.return_value = sample_flight_data.to_dict('records')
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Test ingestion
        result = ingestor.ingest_from_api("https://api.example.com/flights")

        assert result is not None
        assert len(result) == 3
        assert list(result.columns) == list(sample_flight_data.columns)
        mock_get.assert_called_once()

    @patch('requests.get')
    def test_ingest_from_api_failure(self, mock_get, ingestor):
        """Test API ingestion failure."""
        # Mock API failure
        mock_get.side_effect = Exception("API request failed")

        # Test ingestion
        result = ingestor.ingest_from_api("https://api.example.com/flights")

        assert result is None

    @patch('pandas.read_csv')
    def test_ingest_from_file_csv_success(self, mock_read_csv, ingestor, sample_flight_data, tmp_path):
        """Test successful CSV file ingestion."""
        # Create a temporary CSV file
        csv_file = tmp_path / "test_flights.csv"
        sample_flight_data.to_csv(csv_file, index=False)

        # Mock pandas.read_csv
        mock_read_csv.return_value = sample_flight_data

        # Test ingestion
        result = ingestor.ingest_from_file(str(csv_file), "csv")

        assert result is not None
        assert len(result) == 3
        mock_read_csv.assert_called_once()

    @patch('pandas.read_excel')
    def test_ingest_from_file_excel_success(self, mock_read_excel, ingestor, sample_flight_data, tmp_path):
        """Test successful Excel file ingestion."""
        # Create a temporary Excel file
        excel_file = tmp_path / "test_flights.xlsx"
        sample_flight_data.to_excel(excel_file, index=False)

        # Mock pandas.read_excel
        mock_read_excel.return_value = sample_flight_data

        # Test ingestion
        result = ingestor.ingest_from_file(str(excel_file), "excel")

        assert result is not None
        assert len(result) == 3
        mock_read_excel.assert_called_once()

    def test_ingest_from_file_not_found(self, ingestor):
        """Test file ingestion with non-existent file."""
        result = ingestor.ingest_from_file("non_existent_file.csv", "csv")
        assert result is None

    def test_ingest_from_file_unsupported_type(self, ingestor, tmp_path):
        """Test file ingestion with unsupported file type."""
        # Create a temporary file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test data")

        result = ingestor.ingest_from_file(str(test_file), "txt")
        assert result is None

    @patch('pandas.read_sql')
    def test_ingest_flight_schedule(self, mock_read_sql, ingestor, sample_flight_data):
        """Test flight schedule ingestion."""
        # Mock the engine and pandas.read_sql
        ingestor.engine = Mock(spec=Engine)
        mock_read_sql.return_value = sample_flight_data

        # Test ingestion
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        result = ingestor.ingest_flight_schedule(start_date, end_date)

        assert result is not None
        assert len(result) == 3
        mock_read_sql.assert_called_once()

        # Check that the query parameters were passed correctly
        call_args = mock_read_sql.call_args
        assert call_args[1]['params']['start_date'] == start_date
        assert call_args[1]['params']['end_date'] == end_date

    @patch('pandas.read_sql')
    def test_ingest_weather_data(self, mock_read_sql, ingestor, sample_weather_data):
        """Test weather data ingestion."""
        # Mock the engine and pandas.read_sql
        ingestor.engine = Mock(spec=Engine)
        mock_read_sql.return_value = sample_weather_data

        # Test ingestion
        airport_codes = ['LHR', 'JFK', 'CDG']
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        result = ingestor.ingest_weather_data(
            airport_codes, start_date, end_date)

        assert result is not None
        assert len(result) == 3
        mock_read_sql.assert_called_once()

        # Check that the query parameters were passed correctly
        call_args = mock_read_sql.call_args
        assert call_args[1]['params']['airport_codes'] == tuple(airport_codes)
        assert call_args[1]['params']['start_date'] == start_date
        assert call_args[1]['params']['end_date'] == end_date

    @patch('pandas.read_sql')
    def test_ingest_historical_delays(self, mock_read_sql, ingestor):
        """Test historical delay data ingestion."""
        # Create sample delay data
        sample_delay_data = pd.DataFrame({
            'flight_id': ['FL001', 'FL002'],
            'actual_departure': [datetime(2024, 1, 1, 10, 15), datetime(2024, 1, 1, 14, 30)],
            'actual_arrival': [datetime(2024, 1, 1, 13, 30), datetime(2024, 1, 1, 17, 45)],
            'departure_delay': [15, 30],
            'arrival_delay': [30, 45],
            'delay_reason': ['Weather', 'Technical'],
            'weather_conditions': ['Rain', 'Clear']
        })

        # Mock the engine and pandas.read_sql
        ingestor.engine = Mock(spec=Engine)
        mock_read_sql.return_value = sample_delay_data

        # Test ingestion
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        result = ingestor.ingest_historical_delays(start_date, end_date)

        assert result is not None
        assert len(result) == 2
        assert 'departure_delay' in result.columns
        assert 'arrival_delay' in result.columns
        mock_read_sql.assert_called_once()

    def test_ingest_flight_schedule_no_connection(self, ingestor):
        """Test flight schedule ingestion without database connection."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        result = ingestor.ingest_flight_schedule(start_date, end_date)
        assert result is None

    def test_ingest_weather_data_no_connection(self, ingestor):
        """Test weather data ingestion without database connection."""
        airport_codes = ['LHR', 'JFK']
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        result = ingestor.ingest_weather_data(
            airport_codes, start_date, end_date)
        assert result is None

    def test_ingest_historical_delays_no_connection(self, ingestor):
        """Test historical delay ingestion without database connection."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        result = ingestor.ingest_historical_delays(start_date, end_date)
        assert result is None


class TestFlightDataIngestorIntegration:
    """Integration tests for FlightDataIngestor."""

    @pytest.fixture
    def ingestor(self):
        """Create a FlightDataIngestor instance for integration testing."""
        config = Config()
        return FlightDataIngestor(config)

    def test_full_ingestion_workflow(self, ingestor, tmp_path):
        """Test complete ingestion workflow with file-based data."""
        # Create sample data files
        flight_data = pd.DataFrame({
            'flight_id': ['FL001', 'FL002'],
            'origin_airport': ['LHR', 'JFK'],
            'destination_airport': ['JFK', 'LHR'],
            'scheduled_departure': [datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 14, 0)],
            'scheduled_arrival': [datetime(2024, 1, 1, 13, 0), datetime(2024, 1, 1, 17, 0)],
            'aircraft_type': ['A320', 'B737'],
            'airline_code': ['BA', 'AA']
        })

        weather_data = pd.DataFrame({
            'airport_code': ['LHR', 'JFK'],
            'timestamp': [datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 14, 0)],
            'temperature': [15.5, 22.3],
            'humidity': [65, 45],
            'wind_speed': [12.5, 8.2],
            'wind_direction': [180, 270],
            'visibility': [10000, 15000],
            'precipitation': [0.0, 0.0]
        })

        # Save to files
        flight_file = tmp_path / "flights.csv"
        weather_file = tmp_path / "weather.csv"

        flight_data.to_csv(flight_file, index=False)
        weather_data.to_csv(weather_file, index=False)

        # Test ingestion
        flight_result = ingestor.ingest_from_file(str(flight_file), "csv")
        weather_result = ingestor.ingest_from_file(str(weather_file), "csv")

        # Verify results
        assert flight_result is not None
        assert weather_result is not None
        assert len(flight_result) == 2
        assert len(weather_result) == 2
        assert 'flight_id' in flight_result.columns
        assert 'airport_code' in weather_result.columns
