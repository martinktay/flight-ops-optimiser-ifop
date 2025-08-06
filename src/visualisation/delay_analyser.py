"""
Flight delay analysis and visualisation module.

This module provides comprehensive visualisation capabilities for
analysing flight delays, patterns, and operational performance.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class DelayAnalyser:
    """
    A comprehensive flight delay analysis and visualisation system.

    This class provides functionality for creating various charts and
    visualisations to analyse flight delays, patterns, and trends.
    """

    def __init__(self, output_dir: str = "reports/visualisations") -> None:
        """
        Initialise the DelayAnalyser with output directory.

        Args:
            output_dir: Directory to save generated visualisations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up plotting parameters
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12

    def create_delay_summary_dashboard(self, df: pd.DataFrame, save_plot: bool = True) -> plt.Figure:
        """
        Create a comprehensive delay summary dashboard.

        Args:
            df: Flight data DataFrame with delay information
            save_plot: Whether to save the plot to file

        Returns:
            plt.Figure: Generated dashboard figure
        """
        logger.info("Creating delay summary dashboard")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Flight Delay Analysis Dashboard',
                     fontsize=16, fontweight='bold')

        # 1. Delay distribution histogram
        self._plot_delay_distribution(df, axes[0, 0])

        # 2. Delay by day of week
        self._plot_delay_by_day_of_week(df, axes[0, 1])

        # 3. Delay by hour of day
        self._plot_delay_by_hour(df, axes[0, 2])

        # 4. Delay by route
        self._plot_delay_by_route(df, axes[1, 0])

        # 5. Delay by aircraft type
        self._plot_delay_by_aircraft(df, axes[1, 1])

        # 6. Delay trends over time
        self._plot_delay_trends(df, axes[1, 2])

        plt.tight_layout()

        if save_plot:
            self._save_plot(fig, "delay_summary_dashboard.png")

        logger.info("Delay summary dashboard created successfully")
        return fig

    def _plot_delay_distribution(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        """
        Plot delay distribution histogram.

        Args:
            df: Flight data DataFrame
            ax: Matplotlib axes object
        """
        if 'departure_delay' not in df.columns:
            ax.text(0.5, 0.5, 'No delay data available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Delay Distribution')
            return

        delays = df['departure_delay'].dropna()

        # Create bins for different delay categories
        bins = [0, 15, 30, 60, 120, np.inf]
        labels = ['0-15 min', '15-30 min',
                  '30-60 min', '1-2 hours', '2+ hours']

        delay_categories = pd.cut(
            delays, bins=bins, labels=labels, include_lowest=True)
        category_counts = delay_categories.value_counts().sort_index()

        bars = ax.bar(range(len(category_counts)), category_counts.values,
                      color=sns.color_palette("husl", len(category_counts)))

        ax.set_title('Delay Distribution')
        ax.set_xlabel('Delay Duration')
        ax.set_ylabel('Number of Flights')
        ax.set_xticks(range(len(category_counts)))
        ax.set_xticklabels(category_counts.index, rotation=45)

        # Add value labels on bars
        for bar, count in zip(bars, category_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(category_counts.values),
                    f'{count}', ha='center', va='bottom')

    def _plot_delay_by_day_of_week(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        """
        Plot average delay by day of week.

        Args:
            df: Flight data DataFrame
            ax: Matplotlib axes object
        """
        if 'departure_delay' not in df.columns or 'scheduled_departure' not in df.columns:
            ax.text(0.5, 0.5, 'No delay data available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Average Delay by Day of Week')
            return

        # Extract day of week
        df_copy = df.copy()
        df_copy['day_of_week'] = df_copy['scheduled_departure'].dt.day_name()

        # Calculate average delay by day
        day_delays = df_copy.groupby('day_of_week')['departure_delay'].mean().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ])

        bars = ax.bar(range(len(day_delays)), day_delays.values,
                      color=sns.color_palette("husl", len(day_delays)))

        ax.set_title('Average Delay by Day of Week')
        ax.set_xlabel('Day of Week')
        ax.set_ylabel('Average Delay (minutes)')
        ax.set_xticks(range(len(day_delays)))
        ax.set_xticklabels(day_delays.index, rotation=45)

        # Add value labels on bars
        for bar, delay in zip(bars, day_delays.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(day_delays.values),
                    f'{delay:.1f}', ha='center', va='bottom')

    def _plot_delay_by_hour(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        """
        Plot average delay by hour of day.

        Args:
            df: Flight data DataFrame
            ax: Matplotlib axes object
        """
        if 'departure_delay' not in df.columns or 'scheduled_departure' not in df.columns:
            ax.text(0.5, 0.5, 'No delay data available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Average Delay by Hour of Day')
            return

        # Extract hour of day
        df_copy = df.copy()
        df_copy['hour'] = df_copy['scheduled_departure'].dt.hour

        # Calculate average delay by hour
        hour_delays = df_copy.groupby('hour')['departure_delay'].mean()

        ax.plot(hour_delays.index, hour_delays.values,
                marker='o', linewidth=2, markersize=6)
        ax.fill_between(hour_delays.index, hour_delays.values, alpha=0.3)

        ax.set_title('Average Delay by Hour of Day')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Average Delay (minutes)')
        ax.set_xticks(range(0, 24, 2))
        ax.grid(True, alpha=0.3)

    def _plot_delay_by_route(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        """
        Plot average delay by route.

        Args:
            df: Flight data DataFrame
            ax: Matplotlib axes object
        """
        if ('departure_delay' not in df.columns or 'origin_airport' not in df.columns or
                'destination_airport' not in df.columns):
            ax.text(0.5, 0.5, 'No route data available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Average Delay by Route')
            return

        # Create route column
        df_copy = df.copy()
        df_copy['route'] = df_copy['origin_airport'] + \
            ' → ' + df_copy['destination_airport']

        # Calculate average delay by route (top 10)
        route_delays = df_copy.groupby(
            'route')['departure_delay'].mean().sort_values(ascending=False).head(10)

        bars = ax.barh(range(len(route_delays)), route_delays.values,
                       color=sns.color_palette("husl", len(route_delays)))

        ax.set_title('Average Delay by Route (Top 10)')
        ax.set_xlabel('Average Delay (minutes)')
        ax.set_yticks(range(len(route_delays)))
        ax.set_yticklabels(route_delays.index)

        # Add value labels on bars
        for bar, delay in zip(bars, route_delays.values):
            ax.text(bar.get_width() + 0.01*max(route_delays.values), bar.get_y() + bar.get_height()/2,
                    f'{delay:.1f}', ha='left', va='center')

    def _plot_delay_by_aircraft(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        """
        Plot average delay by aircraft type.

        Args:
            df: Flight data DataFrame
            ax: Matplotlib axes object
        """
        if 'departure_delay' not in df.columns or 'aircraft_type' not in df.columns:
            ax.text(0.5, 0.5, 'No aircraft data available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Average Delay by Aircraft Type')
            return

        # Calculate average delay by aircraft type
        aircraft_delays = df.groupby('aircraft_type')[
            'departure_delay'].mean().sort_values(ascending=False)

        # Limit to top 8 for readability
        if len(aircraft_delays) > 8:
            aircraft_delays = aircraft_delays.head(8)

        bars = ax.bar(range(len(aircraft_delays)), aircraft_delays.values,
                      color=sns.color_palette("husl", len(aircraft_delays)))

        ax.set_title('Average Delay by Aircraft Type')
        ax.set_xlabel('Aircraft Type')
        ax.set_ylabel('Average Delay (minutes)')
        ax.set_xticks(range(len(aircraft_delays)))
        ax.set_xticklabels(aircraft_delays.index, rotation=45)

        # Add value labels on bars
        for bar, delay in zip(bars, aircraft_delays.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(aircraft_delays.values),
                    f'{delay:.1f}', ha='center', va='bottom')

    def _plot_delay_trends(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        """
        Plot delay trends over time.

        Args:
            df: Flight data DataFrame
            ax: Matplotlib axes object
        """
        if 'departure_delay' not in df.columns or 'scheduled_departure' not in df.columns:
            ax.text(0.5, 0.5, 'No delay data available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Delay Trends Over Time')
            return

        # Resample by day and calculate average delay
        df_copy = df.copy()
        df_copy.set_index('scheduled_departure', inplace=True)

        daily_delays = df_copy['departure_delay'].resample('D').mean()

        # Plot trend line
        ax.plot(daily_delays.index, daily_delays.values, linewidth=2, alpha=0.7)
        ax.fill_between(daily_delays.index, daily_delays.values, alpha=0.3)

        # Add moving average
        if len(daily_delays) > 7:
            moving_avg = daily_delays.rolling(window=7).mean()
            ax.plot(moving_avg.index, moving_avg.values, linewidth=2, color='red',
                    label='7-day Moving Average')
            ax.legend()

        ax.set_title('Delay Trends Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Average Delay (minutes)')
        ax.grid(True, alpha=0.3)

        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45)

    def create_weather_delay_analysis(self, df: pd.DataFrame, save_plot: bool = True) -> plt.Figure:
        """
        Create weather-related delay analysis.

        Args:
            df: Flight data DataFrame with weather information
            save_plot: Whether to save the plot to file

        Returns:
            plt.Figure: Generated weather analysis figure
        """
        logger.info("Creating weather delay analysis")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Weather Impact on Flight Delays',
                     fontsize=16, fontweight='bold')

        # 1. Delay by temperature
        self._plot_delay_by_temperature(df, axes[0, 0])

        # 2. Delay by wind speed
        self._plot_delay_by_wind_speed(df, axes[0, 1])

        # 3. Delay by visibility
        self._plot_delay_by_visibility(df, axes[1, 0])

        # 4. Delay by precipitation
        self._plot_delay_by_precipitation(df, axes[1, 1])

        plt.tight_layout()

        if save_plot:
            self._save_plot(fig, "weather_delay_analysis.png")

        logger.info("Weather delay analysis created successfully")
        return fig

    def _plot_delay_by_temperature(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot delay by temperature."""
        if 'departure_delay' not in df.columns or 'temperature' not in df.columns:
            ax.text(0.5, 0.5, 'No temperature data available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Delay by Temperature')
            return

        # Create temperature bins
        df_copy = df.copy()
        df_copy['temp_bin'] = pd.cut(df_copy['temperature'], bins=10)
        temp_delays = df_copy.groupby('temp_bin')['departure_delay'].mean()

        ax.bar(range(len(temp_delays)), temp_delays.values,
               color=sns.color_palette("coolwarm", len(temp_delays)))
        ax.set_title('Average Delay by Temperature')
        ax.set_xlabel('Temperature Range (°C)')
        ax.set_ylabel('Average Delay (minutes)')
        ax.set_xticks(range(len(temp_delays)))
        ax.set_xticklabels(
            [f'{interval.left:.0f}°C' for interval in temp_delays.index], rotation=45)

    def _plot_delay_by_wind_speed(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot delay by wind speed."""
        if 'departure_delay' not in df.columns or 'wind_speed' not in df.columns:
            ax.text(0.5, 0.5, 'No wind speed data available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Delay by Wind Speed')
            return

        # Create wind speed bins
        df_copy = df.copy()
        df_copy['wind_bin'] = pd.cut(df_copy['wind_speed'], bins=8)
        wind_delays = df_copy.groupby('wind_bin')['departure_delay'].mean()

        ax.bar(range(len(wind_delays)), wind_delays.values,
               color=sns.color_palette("Blues", len(wind_delays)))
        ax.set_title('Average Delay by Wind Speed')
        ax.set_xlabel('Wind Speed Range (km/h)')
        ax.set_ylabel('Average Delay (minutes)')
        ax.set_xticks(range(len(wind_delays)))
        ax.set_xticklabels(
            [f'{interval.left:.0f} km/h' for interval in wind_delays.index], rotation=45)

    def _plot_delay_by_visibility(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot delay by visibility."""
        if 'departure_delay' not in df.columns or 'visibility' not in df.columns:
            ax.text(0.5, 0.5, 'No visibility data available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Delay by Visibility')
            return

        # Create visibility bins
        df_copy = df.copy()
        df_copy['vis_bin'] = pd.cut(df_copy['visibility'], bins=6)
        vis_delays = df_copy.groupby('vis_bin')['departure_delay'].mean()

        ax.bar(range(len(vis_delays)), vis_delays.values,
               color=sns.color_palette("Greys", len(vis_delays)))
        ax.set_title('Average Delay by Visibility')
        ax.set_xlabel('Visibility Range (m)')
        ax.set_ylabel('Average Delay (minutes)')
        ax.set_xticks(range(len(vis_delays)))
        ax.set_xticklabels(
            [f'{interval.left:.0f}m' for interval in vis_delays.index], rotation=45)

    def _plot_delay_by_precipitation(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot delay by precipitation."""
        if 'departure_delay' not in df.columns or 'precipitation' not in df.columns:
            ax.text(0.5, 0.5, 'No precipitation data available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Delay by Precipitation')
            return

        # Create precipitation bins
        df_copy = df.copy()
        df_copy['precip_bin'] = pd.cut(df_copy['precipitation'], bins=5)
        precip_delays = df_copy.groupby('precip_bin')['departure_delay'].mean()

        ax.bar(range(len(precip_delays)), precip_delays.values,
               color=sns.color_palette("Blues", len(precip_delays)))
        ax.set_title('Average Delay by Precipitation')
        ax.set_xlabel('Precipitation Range (mm)')
        ax.set_ylabel('Average Delay (minutes)')
        ax.set_xticks(range(len(precip_delays)))
        ax.set_xticklabels(
            [f'{interval.left:.1f}mm' for interval in precip_delays.index], rotation=45)

    def create_operational_performance_report(self, df: pd.DataFrame, save_plot: bool = True) -> plt.Figure:
        """
        Create operational performance report.

        Args:
            df: Flight data DataFrame
            save_plot: Whether to save the plot to file

        Returns:
            plt.Figure: Generated performance report figure
        """
        logger.info("Creating operational performance report")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Operational Performance Analysis',
                     fontsize=16, fontweight='bold')

        # 1. On-time performance
        self._plot_on_time_performance(df, axes[0, 0])

        # 2. Delay reasons breakdown
        self._plot_delay_reasons(df, axes[0, 1])

        # 3. Flight duration vs delay
        self._plot_duration_vs_delay(df, axes[1, 0])

        # 4. Monthly performance trends
        self._plot_monthly_performance(df, axes[1, 1])

        plt.tight_layout()

        if save_plot:
            self._save_plot(fig, "operational_performance_report.png")

        logger.info("Operational performance report created successfully")
        return fig

    def _plot_on_time_performance(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot on-time performance metrics."""
        if 'departure_delay' not in df.columns:
            ax.text(0.5, 0.5, 'No delay data available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('On-Time Performance')
            return

        delays = df['departure_delay'].dropna()

        # Calculate performance metrics
        on_time = (delays <= 0).sum()
        minor_delays = ((delays > 0) & (delays <= 15)).sum()
        moderate_delays = ((delays > 15) & (delays <= 60)).sum()
        significant_delays = (delays > 60).sum()

        categories = [
            'On Time', 'Minor Delays\n(0-15 min)', 'Moderate Delays\n(15-60 min)', 'Significant Delays\n(>60 min)']
        values = [on_time, minor_delays, moderate_delays, significant_delays]
        colors = ['green', 'yellow', 'orange', 'red']

        wedges, texts, autotexts = ax.pie(
            values, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('On-Time Performance Breakdown')

        # Make text more readable
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

    def _plot_delay_reasons(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot delay reasons breakdown."""
        if 'delay_reason' not in df.columns:
            ax.text(0.5, 0.5, 'No delay reason data available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Delay Reasons')
            return

        reason_counts = df['delay_reason'].value_counts().head(8)

        bars = ax.bar(range(len(reason_counts)), reason_counts.values,
                      color=sns.color_palette("Set3", len(reason_counts)))

        ax.set_title('Top Delay Reasons')
        ax.set_xlabel('Delay Reason')
        ax.set_ylabel('Number of Flights')
        ax.set_xticks(range(len(reason_counts)))
        ax.set_xticklabels(reason_counts.index, rotation=45, ha='right')

        # Add value labels on bars
        for bar, count in zip(bars, reason_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(reason_counts.values),
                    f'{count}', ha='center', va='bottom')

    def _plot_duration_vs_delay(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot flight duration vs delay scatter plot."""
        if ('departure_delay' not in df.columns or 'scheduled_departure' not in df.columns or
                'scheduled_arrival' not in df.columns):
            ax.text(0.5, 0.5, 'No duration data available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Flight Duration vs Delay')
            return

        # Calculate flight duration
        df_copy = df.copy()
        df_copy['flight_duration'] = (
            df_copy['scheduled_arrival'] - df_copy['scheduled_departure']).dt.total_seconds() / 60

        # Create scatter plot
        ax.scatter(df_copy['flight_duration'],
                   df_copy['departure_delay'], alpha=0.6, s=20)

        # Add trend line
        z = np.polyfit(df_copy['flight_duration'].dropna(),
                       df_copy['departure_delay'].dropna(), 1)
        p = np.poly1d(z)
        ax.plot(df_copy['flight_duration'], p(
            df_copy['flight_duration']), "r--", alpha=0.8)

        ax.set_title('Flight Duration vs Delay')
        ax.set_xlabel('Flight Duration (minutes)')
        ax.set_ylabel('Departure Delay (minutes)')
        ax.grid(True, alpha=0.3)

    def _plot_monthly_performance(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        """Plot monthly performance trends."""
        if 'departure_delay' not in df.columns or 'scheduled_departure' not in df.columns:
            ax.text(0.5, 0.5, 'No monthly data available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Monthly Performance Trends')
            return

        # Extract month and calculate metrics
        df_copy = df.copy()
        df_copy['month'] = df_copy['scheduled_departure'].dt.to_period('M')

        monthly_stats = df_copy.groupby('month').agg({
            'departure_delay': ['mean', 'std', 'count']
        }).round(2)

        months = [str(period) for period in monthly_stats.index]
        avg_delays = monthly_stats[('departure_delay', 'mean')].values

        ax.plot(range(len(months)), avg_delays,
                marker='o', linewidth=2, markersize=6)
        ax.fill_between(range(len(months)), avg_delays, alpha=0.3)

        ax.set_title('Monthly Average Delay Trends')
        ax.set_xlabel('Month')
        ax.set_ylabel('Average Delay (minutes)')
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels(months, rotation=45)
        ax.grid(True, alpha=0.3)

    def _save_plot(self, fig: plt.Figure, filename: str) -> None:
        """
        Save plot to file.

        Args:
            fig: Matplotlib figure object
            filename: Output filename
        """
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {filepath}")

    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for delay analysis.

        Args:
            df: Flight data DataFrame

        Returns:
            Dict[str, Any]: Summary statistics
        """
        if 'departure_delay' not in df.columns:
            return {'error': 'No delay data available'}

        delays = df['departure_delay'].dropna()

        stats = {
            'total_flights': len(df),
            'flights_with_delays': len(delays),
            'on_time_percentage': ((delays <= 0).sum() / len(delays) * 100) if len(delays) > 0 else 0,
            'average_delay': delays.mean(),
            'median_delay': delays.median(),
            'max_delay': delays.max(),
            'delay_std': delays.std(),
            'delay_categories': {
                'on_time': (delays <= 0).sum(),
                'minor_0_15': ((delays > 0) & (delays <= 15)).sum(),
                'moderate_15_60': ((delays > 15) & (delays <= 60)).sum(),
                'significant_60_120': ((delays > 60) & (delays <= 120)).sum(),
                'major_120_plus': (delays > 120).sum()
            }
        }

        logger.info(f"Generated summary statistics for {len(df)} flights")
        return stats
