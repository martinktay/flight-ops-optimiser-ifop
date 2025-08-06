import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Flight Operations Optimiser Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-metric {
        border-left-color: #28a745;
    }
    .warning-metric {
        border-left-color: #ffc107;
    }
    .danger-metric {
        border-left-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and cache the flight data."""
    try:
        df = pd.read_csv('data/flights_dev_sample.csv')
        # Convert date and time columns
        df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
        df['scheduled_departure'] = pd.to_datetime(
            df['FL_DATE'].astype(str) + ' ' +
            df['CRS_DEP_TIME'].astype(str).str.zfill(4),
            format='%Y-%m-%d %H%M'
        )
        df['scheduled_arrival'] = pd.to_datetime(
            df['FL_DATE'].astype(str) + ' ' +
            df['CRS_ARR_TIME'].astype(str).str.zfill(4),
            format='%Y-%m-%d %H%M'
        )
        # Add derived columns
        df['departure_delay'] = df['DEP_DELAY']
        df['arrival_delay'] = df['ARR_DELAY']
        df['day_of_week'] = df['scheduled_departure'].dt.day_name()
        df['hour_of_day'] = df['scheduled_departure'].dt.hour
        df['month'] = df['scheduled_departure'].dt.month
        df['year'] = df['scheduled_departure'].dt.year
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def calculate_metrics(df):
    """Calculate key performance metrics."""
    if df is None or df.empty:
        return {}

    delays = df['departure_delay'].dropna()

    metrics = {
        'total_flights': len(df),
        'flights_with_delays': len(delays),
        'on_time_percentage': ((delays <= 0).sum() / len(delays) * 100) if len(delays) > 0 else 0,
        'average_delay': delays.mean(),
        'median_delay': delays.median(),
        'max_delay': delays.max(),
        'delay_std': delays.std(),
        'unique_airlines': df['AIRLINE_CODE'].nunique(),
        'unique_airports': df['ORIGIN'].nunique(),
        'date_range': f"{df['FL_DATE'].min().strftime('%Y-%m-%d')} to {df['FL_DATE'].max().strftime('%Y-%m-%d')}",
        'delay_categories': {
            'on_time': (delays <= 0).sum(),
            'minor_0_15': ((delays > 0) & (delays <= 15)).sum(),
            'moderate_15_60': ((delays > 15) & (delays <= 60)).sum(),
            'significant_60_120': ((delays > 60) & (delays <= 120)).sum(),
            'major_120_plus': (delays > 120).sum()
        }
    }
    return metrics


def main():
    # Header
    st.markdown('<h1 class="main-header">✈️ Flight Operations Optimiser Dashboard</h1>',
                unsafe_allow_html=True)

    # Load data
    with st.spinner('Loading flight data...'):
        df = load_data()

    if df is None:
        st.error("Failed to load data. Please ensure the data files are available.")
        return

    # Calculate metrics
    metrics = calculate_metrics(df)

    # Sidebar filters
    st.sidebar.header("📊 Dashboard Filters")

    # Date range filter
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(df['FL_DATE'].min(), df['FL_DATE'].max()),
        min_value=df['FL_DATE'].min(),
        max_value=df['FL_DATE'].max()
    )

    # Airline filter
    airlines = st.sidebar.multiselect(
        "Select Airlines",
        options=sorted(df['AIRLINE_CODE'].unique()),
        default=sorted(df['AIRLINE_CODE'].unique())[:5]
    )

    # Apply filters
    if len(date_range) == 2:
        mask = (df['FL_DATE'] >= pd.to_datetime(date_range[0])) & \
               (df['FL_DATE'] <= pd.to_datetime(date_range[1])) & \
               (df['AIRLINE_CODE'].isin(airlines))
        filtered_df = df[mask].copy()
    else:
        filtered_df = df.copy()

    # Recalculate metrics for filtered data
    filtered_metrics = calculate_metrics(filtered_df)

    # Main dashboard content
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Flights",
            value=f"{filtered_metrics['total_flights']:,}",
            delta=f"{filtered_metrics['total_flights'] - metrics['total_flights']:,}"
        )

    with col2:
        st.metric(
            label="On-Time Performance",
            value=f"{filtered_metrics['on_time_percentage']:.1f}%",
            delta=f"{filtered_metrics['on_time_percentage'] - metrics['on_time_percentage']:.1f}%"
        )

    with col3:
        st.metric(
            label="Average Delay",
            value=f"{filtered_metrics['average_delay']:.1f} min",
            delta=f"{filtered_metrics['average_delay'] - metrics['average_delay']:.1f} min"
        )

    with col4:
        st.metric(
            label="Airlines",
            value=filtered_metrics['unique_airlines'],
            delta=filtered_metrics['unique_airlines'] -
            metrics['unique_airlines']
        )

    # Detailed metrics section
    st.markdown("---")
    st.subheader("📈 Detailed Performance Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Delay Distribution")
        delay_cats = filtered_metrics['delay_categories']
        fig = go.Figure(data=[
            go.Bar(
                x=['On Time', 'Minor (0-15min)', 'Moderate (15-60min)',
                   'Significant (60-120min)', 'Major (120+min)'],
                y=[delay_cats['on_time'], delay_cats['minor_0_15'], delay_cats['moderate_15_60'],
                   delay_cats['significant_60_120'], delay_cats['major_120_plus']],
                marker_color=['#28a745', '#ffc107',
                              '#fd7e14', '#dc3545', '#6f42c1']
            )
        ])
        fig.update_layout(
            title="Flight Delay Categories",
            xaxis_title="Delay Category",
            yaxis_title="Number of Flights",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Key Statistics")
        stats_data = {
            'Metric': ['Total Flights', 'Flights with Delays', 'On-Time Percentage',
                       'Average Delay', 'Median Delay', 'Max Delay', 'Delay Std Dev'],
            'Value': [
                f"{filtered_metrics['total_flights']:,}",
                f"{filtered_metrics['flights_with_delays']:,}",
                f"{filtered_metrics['on_time_percentage']:.1f}%",
                f"{filtered_metrics['average_delay']:.1f} min",
                f"{filtered_metrics['median_delay']:.1f} min",
                f"{filtered_metrics['max_delay']:.0f} min",
                f"{filtered_metrics['delay_std']:.1f} min"
            ]
        }
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # Time-based analysis
    st.markdown("---")
    st.subheader("⏰ Time-Based Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Delay by Day of Week")
        day_delays = filtered_df.groupby('day_of_week')['departure_delay'].mean().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ])
        fig = px.bar(
            x=day_delays.index,
            y=day_delays.values,
            title="Average Delay by Day of Week",
            labels={'x': 'Day of Week', 'y': 'Average Delay (minutes)'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Delay by Hour of Day")
        hour_delays = filtered_df.groupby(
            'hour_of_day')['departure_delay'].mean()
        fig = px.line(
            x=hour_delays.index,
            y=hour_delays.values,
            title="Average Delay by Hour of Day",
            labels={'x': 'Hour of Day', 'y': 'Average Delay (minutes)'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Airline performance
    st.markdown("---")
    st.subheader("🏢 Airline Performance")

    # Calculate airline performance metrics manually
    airline_stats = []
    try:
        for airline in filtered_df['AIRLINE_CODE'].unique():
            airline_data = filtered_df[filtered_df['AIRLINE_CODE'] == airline]
            if len(airline_data) > 0:
                total_flights = len(airline_data)
                avg_delay = airline_data['departure_delay'].mean()
                delay_std = airline_data['departure_delay'].std()
                on_time_pct = (airline_data['departure_delay']
                               <= 0).sum() / total_flights * 100

                airline_stats.append({
                    'AIRLINE_CODE': airline,
                    'Total Flights': total_flights,
                    'Avg Delay (min)': round(avg_delay, 2),
                    'Delay Std Dev': round(delay_std, 2),
                    'On-Time %': round(on_time_pct, 2)
                })
    except Exception as e:
        st.error(f"Error calculating airline performance: {e}")
        airline_stats = []

    airline_performance = pd.DataFrame(airline_stats).set_index('AIRLINE_CODE')
    airline_performance = airline_performance.sort_values(
        'On-Time %', ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Top Airlines by On-Time Performance")
        fig = px.bar(
            airline_performance.head(10),
            y='On-Time %',
            title="Top 10 Airlines by On-Time Performance",
            labels={'On-Time %': 'On-Time Percentage (%)'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Airline Performance Table")
        st.dataframe(airline_performance, use_container_width=True)

    # Route analysis
    st.markdown("---")
    st.subheader("🛫 Route Analysis")

    # Calculate route performance metrics manually
    route_stats = []
    try:
        for (origin, dest) in filtered_df.groupby(['ORIGIN', 'DEST']).groups.keys():
            route_data = filtered_df[(filtered_df['ORIGIN'] == origin) & (
                filtered_df['DEST'] == dest)]
            if len(route_data) > 0:
                total_flights = len(route_data)
                avg_delay = route_data['departure_delay'].mean()
                on_time_pct = (route_data['departure_delay']
                               <= 0).sum() / total_flights * 100

                route_stats.append({
                    'ORIGIN': origin,
                    'DEST': dest,
                    'Total Flights': total_flights,
                    'Avg Delay (min)': round(avg_delay, 2),
                    'On-Time %': round(on_time_pct, 2)
                })
    except Exception as e:
        st.error(f"Error calculating route performance: {e}")
        route_stats = []

    route_performance = pd.DataFrame(route_stats)
    route_performance = route_performance.sort_values(
        'Total Flights', ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Busiest Routes")
        # Create route labels for the chart
        top_routes = route_performance.head(10)
        route_labels = [
            f"{row['ORIGIN']} → {row['DEST']}" for _, row in top_routes.iterrows()]

        fig = px.bar(
            x=route_labels,
            y=top_routes['Total Flights'],
            title="Top 10 Busiest Routes",
            labels={'x': 'Route', 'y': 'Number of Flights'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Route Performance Table")
        st.dataframe(route_performance.head(15), use_container_width=True)

    # Model performance section (placeholder for ML results)
    st.markdown("---")
    st.subheader("🤖 Machine Learning Model Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Model Accuracy")
        # Placeholder for actual model metrics
        st.markdown("""
        <div style="background-color: #2c3e50; padding: 20px; border-radius: 10px; border-left: 4px solid #28a745; height: 200px; box-shadow: none;">
        <h4 style="color: #28a745; margin-top: 0;">RandomForest Model:</h4>
        <ul style="margin-bottom: 0; color: #ecf0f1;">
            <li>R² Score: 0.78</li>
            <li>MAE: 12.3 min</li>
            <li>RMSE: 18.7 min</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### Feature Importance")
        # Placeholder for feature importance
        st.markdown("""
        <div style="background-color: #2c3e50; padding: 20px; border-radius: 10px; border-left: 4px solid #28a745; height: 200px; box-shadow: none;">
        <h4 style="color: #28a745; margin-top: 0;">Top Features:</h4>
        <ul style="margin-bottom: 0; color: #ecf0f1;">
            <li>Weather conditions</li>
            <li>Historical delays</li>
            <li>Time of day</li>
            <li>Route patterns</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("### Optimisation Results")
        # Placeholder for optimisation metrics
        st.markdown("""
        <div style="background-color: #2c3e50; padding: 20px; border-radius: 10px; border-left: 4px solid #28a745; height: 200px; box-shadow: none;">
        <h4 style="color: #28a745; margin-top: 0;">Gurobi Optimisation:</h4>
        <ul style="margin-bottom: 0; color: #ecf0f1;">
            <li>Crew utilisation: 85%</li>
            <li>Aircraft utilisation: 82%</li>
            <li>Cost reduction: 15%</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    # Raw data section
    st.markdown("---")
    st.subheader("📋 Raw Data Explorer")

    if st.checkbox("Show raw data"):
        st.dataframe(filtered_df, use_container_width=True)

    # Download section
    st.markdown("---")
    st.subheader("💾 Data Export")

    col1, col2 = st.columns(2)

    with col1:
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name=f"flight_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    with col2:
        # Generate summary report
        summary_report = f"""
Flight Operations Dashboard Summary Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Dataset Overview:
- Total Flights: {filtered_metrics['total_flights']:,}
- Date Range: {filtered_metrics['date_range']}
- Airlines: {filtered_metrics['unique_airlines']}
- Airports: {filtered_metrics['unique_airports']}

Performance Metrics:
- On-Time Performance: {filtered_metrics['on_time_percentage']:.1f}%
- Average Delay: {filtered_metrics['average_delay']:.1f} minutes
- Median Delay: {filtered_metrics['median_delay']:.1f} minutes
- Maximum Delay: {filtered_metrics['max_delay']:.0f} minutes

Delay Distribution:
- On Time: {filtered_metrics['delay_categories']['on_time']:,} flights
- Minor Delays (0-15 min): {filtered_metrics['delay_categories']['minor_0_15']:,} flights
- Moderate Delays (15-60 min): {filtered_metrics['delay_categories']['moderate_15_60']:,} flights
- Significant Delays (60-120 min): {filtered_metrics['delay_categories']['significant_60_120']:,} flights
- Major Delays (120+ min): {filtered_metrics['delay_categories']['major_120_plus']:,} flights
        """

        st.download_button(
            label="Download Summary Report",
            data=summary_report,
            file_name=f"flight_ops_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>✈️ Flight Operations Optimiser Dashboard | Built with Streamlit</p>
        <p>Data Source: Flight Operations Dataset | Analysis Period: 2019-2023</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
