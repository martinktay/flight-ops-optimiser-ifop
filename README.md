# Flight Operations Optimiser (IFOP)

A modular Python project that predicts flight delays and optimises crew/aircraft scheduling using machine learning and linear programming.

## Features

- **Flight Delay Prediction**: Machine learning models to predict flight delays
- **Crew & Aircraft Optimisation**: Linear programming optimisation using Gurobi
- **Data Pipeline**: Automated data ingestion and cleaning using Dagster
- **Visualisation**: Interactive dashboards and delay analysis charts
- **ML Tracking**: Experiment tracking and model versioning with MLflow
- **Testing**: Comprehensive unit tests with pytest
- **CI/CD**: Automated testing and deployment with GitHub Actions
- **Containerisation**: Docker support for easy deployment

## Project Structure

```
flight-ops-optimiser-ifop/
├── src/
│   ├── data/
│   │   ├── ingestion/
│   │   ├── cleaning/
│   │   └── validation/
│   ├── models/
│   │   ├── delay_prediction/
│   │   └── optimisation/
│   ├── visualisation/
│   └── utils/
├── tests/
├── configs/
├── data/
├── notebooks/
├── docker/
└── .github/
```

## Installation

### Prerequisites

- Python 3.9+
- Gurobi Optimiser (requires licence)
- Docker (optional)

### Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd flight-ops-optimiser-ifop
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:

```bash
cp .env.example .env
# Edit .env with your configuration
```

## Dataset Integration

This section explains how the flight dataset has been integrated into the Intelligent Flight Operations Optimiser (iFOP) project.

### Dataset Overview

The project originally included a `flights_sample_3m.csv` file containing **3 million+ flight records** with comprehensive flight operations data covering the period from **2019-01-01 to 2023-08-31**. To make the project more manageable and reduce repository size, this large file has been removed and replaced with pre-generated sample files.

#### Key Features

- **Flight Information**: Date, airline, flight number, origin/destination airports
- **Timing Data**: Scheduled vs actual departure/arrival times
- **Delay Analysis**: Departure and arrival delays with causal breakdown
- **Operational Metrics**: Distance, cancellations, diversions
- **Advanced Features**: Taxi times, air time, elapsed time

#### Dataset Statistics

- **Total Records**: 3,000,000+
- **Date Range**: 2019-01-01 to 2023-08-31 (1,704 unique dates)
- **Airlines**: 18+ different carriers
- **Airports**: 360+ origin and destination airports
- **Columns**: 32 total columns (21 essential + 7 optional)

### Dataset Filtering System

To make this large dataset manageable for development and analysis, we've implemented a comprehensive filtering system.

#### Core Components

1. **`DatasetFilter` Class** (`src/data/preprocessing/dataset_filter.py`)

   - Handles all dataset filtering and sampling operations
   - Provides multiple filtering options
   - Generates summary statistics and reports

2. **Enhanced Data Ingestion** (`src/data/ingestion/flight_data_ingestor.py`)

   - New `ingest_flights_dataset()` method
   - Integrated with existing Dagster pipeline
   - Supports filtering during ingestion

3. **Makefile Integration**
   - Convenient commands for common filtering tasks
   - Automated sample generation

#### Filtering Options

##### 1. Date Range Filtering

```python
# Filter by specific date range
df = filter_tool.generate_filtered_dataset(
    output_file="data/flights_jan_2023.csv",
    date_range=("2023-01-01", "2023-01-31")
)
```

##### 2. Airline Filtering

```python
# Filter by specific airlines
df = filter_tool.generate_filtered_dataset(
    output_file="data/flights_major_airlines.csv",
    airline_codes=['AA', 'DL', 'UA', 'WN']  # American, Delta, United, Southwest
)
```

##### 3. Random Sampling

```python
# Take a random sample
df = filter_tool.generate_filtered_dataset(
    output_file="data/flights_sample.csv",
    sample_size=50000
)
```

##### 4. Column Selection

```python
# Include optional columns for advanced analysis
df = filter_tool.generate_filtered_dataset(
    output_file="data/flights_full_featured.csv",
    include_optional_columns=True
)
```

### Pre-built Samples

The system generates several pre-configured samples for different use cases:

#### 1. Development Sample (`flights_dev_sample.csv`)

- **Size**: 50,000 rows
- **Columns**: Essential columns only (21 columns)
- **Use Case**: Quick development and testing
- **Command**: `make create-dev-sample`

#### 2. Major Airlines Sample (`flights_major_airlines.csv`)

- **Size**: 100,000 rows
- **Airlines**: AA, DL, UA, WN, AS
- **Use Case**: Airline-specific analysis
- **Command**: `make create-airline-sample`

#### 3. Recent Data Sample (`flights_recent_2023.csv`)

- **Size**: 75,000 rows
- **Period**: June-August 2023 (last 3 months of available data)
- **Use Case**: Recent trends analysis
- **Command**: `make create-recent-sample`

#### 4. Full Featured Sample (`flights_full_featured.csv`)

- **Size**: 25,000 rows
- **Columns**: All columns (28 columns)
- **Use Case**: Detailed analysis with all features
- **Command**: `make filter-dataset`

### Usage Examples

#### Quick Start

1. **Generate all samples**:

   ```bash
   make filter-dataset
   ```

2. **Create a specific sample**:

   ```bash
   make create-dev-sample
   ```

3. **Use in Python**:

   ```python
   from src.data.preprocessing.dataset_filter import DatasetFilter

   # Create filter instance
   filter_tool = DatasetFilter()

   # Generate custom sample
   df = filter_tool.generate_filtered_dataset(
       output_file="data/my_sample.csv",
       sample_size=10000,
       airline_codes=['AA', 'DL'],
       date_range=("2023-01-01", "2023-03-31")
   )

   # Get summary statistics
   stats = filter_tool.generate_summary_statistics(df)
   filter_tool.print_summary_report(stats)
   ```

#### Integration with Data Pipeline

The dataset is integrated into the Dagster pipeline:

```python
# In src/dagster_pipeline.py
@asset(description="Raw historical delay data for model training")
def raw_delay_data(context: AssetExecutionContext, config: FlightOpsConfig) -> pd.DataFrame:
    # Uses the new ingest_flights_dataset method with generated sample
    delay_data = ingestor.ingest_flights_dataset(
        file_path="data/flights_dev_sample.csv",  # Use generated development sample
        sample_size=None  # No additional sampling needed
    )
    return delay_data
```

### Performance Considerations

#### Memory Usage

- **Original dataset**: ~586MB on disk, ~2-3GB in memory (removed from repository)
- **50K sample**: ~5MB on disk, ~50-100MB in memory
- **100K sample**: ~10MB on disk, ~100-200MB in memory

#### Processing Time

- **Original dataset loading**: ~30-45 seconds (no longer available)
- **50K sample loading**: ~5-10 seconds
- **100K sample loading**: ~10-15 seconds

#### Recommendations

- Use 50K-100K samples for development and testing
- Use larger samples for production training if needed
- Consider date range filtering for specific analysis periods
- Generate new samples using the filtering utilities if required

### File Structure

```
flight-ops-optimiser-ifop/
├── data/                          # Generated samples
│   ├── flights_dev_sample.csv     # 50K rows, essential columns
│   ├── flights_major_airlines.csv # 100K rows, major airlines
│   ├── flights_recent_2023.csv    # 75K rows, recent data
│   └── flights_full_featured.csv  # 25K rows, all columns
├── src/data/preprocessing/
│   └── dataset_filter.py          # Main filtering logic
└── scripts/
    └── test_dataset_filter.py     # Test and demo script
```

### Troubleshooting

#### Common Issues

1. **Memory Error**: Reduce sample size or use date range filtering
2. **File Not Found**: Ensure the required sample files are in the `data/` directory
3. **Date Range Returns 0 Rows**: Check available date range (2019-01-01 to 2023-08-31)

#### Performance Tips

1. **For Development**: Use `flights_dev_sample.csv` (50K rows)
2. **For Testing**: Use `flights_major_airlines.csv` (100K rows)
3. **For Production**: Use full dataset with appropriate filtering

### Next Steps

1. **Model Training**: Use the filtered samples to train delay prediction models
2. **Optimisation**: Use the data for crew and aircraft scheduling optimisation
3. **Analysis**: Generate delay analysis reports and visualisations
4. **Production**: Use larger samples or generate new ones for production deployment

The dataset integration provides a solid foundation for all flight operations analysis and optimisation tasks in the iFOP project. The removal of the large original file makes the repository more manageable while maintaining all functionality through the pre-generated samples.

## Usage

### Running the Data Pipeline

```bash
dagster dev
```

### Training Models

```bash
python -m src.models.delay_prediction.train
```

### Running Optimisation

```bash
python -m src.models.optimisation.scheduler
```

### Running Tests

```bash
pytest tests/
```

## Configuration

The project uses configuration files in the `configs/` directory:

- `data_config.yaml`: Data source and processing settings
- `model_config.yaml`: Model hyperparameters and training settings
- `optimisation_config.yaml`: Optimisation constraints and objectives

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## Licence

This project is licensed under the MIT Licence - see the [LICENCE](LICENCE) file for details.

## Support

For support and questions, please open an issue on GitHub or contact the development team.
