import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

class TemperatureAnalyzer:
    def __init__(self, data_path):
        """Initialize with the path to the temperature logs CSV file."""
        self.data_path = data_path
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load the CSV data and perform initial processing."""
        # Load the CSV file
        self.df = pd.read_csv(self.data_path)
        
        # Convert timestamp to datetime
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], format='%Y/%m/%d %H:%M')
        
        # Convert temperature strings to float, handling missing values
        self.df['temperature_c'] = pd.to_numeric(self.df['temperature_c'], errors='coerce')
        
        # Sort by timestamp
        self.df = self.df.sort_values('timestamp')
        
        print(f"Loaded {len(self.df)} records from {self.data_path}")
        
    def get_temperature_stats(self):
        """Return basic statistics about the temperature data."""
        temp_stats = {
            'min_temp': self.df['temperature_c'].min(),
            'max_temp': self.df['temperature_c'].max(),
            'avg_temp': self.df['temperature_c'].mean(),
            'median_temp': self.df['temperature_c'].median(),
            'missing_values': self.df['temperature_c'].isna().sum()
        }
        return temp_stats
    
    def get_temperature_by_location(self):
        """Group temperature readings by location."""
        return self.df.groupby('location')['temperature_c'].agg(['mean', 'min', 'max', 'count'])
    
    def plot_temperature_trends(self, output_path='../reports/temperature_trends.png'):
        """Plot temperature trends over time and save the figure."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Filter rows with valid temperature readings
        temp_data = self.df.dropna(subset=['temperature_c'])
        
        if len(temp_data) == 0:
            print("No valid temperature data available for plotting.")
            return
            
        # Create the plot
        plt.figure(figsize=(12, 6))
        
        # Group by location for different colored lines
        for location, group in temp_data.groupby('location'):
            plt.plot(group['timestamp'], group['temperature_c'], marker='o', linestyle='-', label=location)
        
        plt.title('Temperature Trends by Location')
        plt.xlabel('Date and Time')
        plt.ylabel('Temperature (°C)')
        plt.grid(True)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(output_path)
        print(f"Temperature trend plot saved to {output_path}")
        
    def detect_anomalies(self, threshold=2.0):
        """Detect temperature readings that are outside normal range.
        
        Args:
            threshold: Number of standard deviations from the mean to consider anomalous
        
        Returns:
            DataFrame containing anomalous readings
        """
        if self.df['temperature_c'].isna().all():
            print("No temperature data available for anomaly detection.")
            return pd.DataFrame()
            
        # Calculate mean and standard deviation for each location
        location_stats = self.df.groupby('location')['temperature_c'].agg(['mean', 'std'])
        
        # Initialize an empty list to store anomalies
        anomalies = []
        
        # Check each location
        for location, group in self.df.groupby('location'):
            if location not in location_stats.index or pd.isna(location_stats.loc[location, 'std']):
                continue
                
            mean = location_stats.loc[location, 'mean']
            std = location_stats.loc[location, 'std']
            
            # Find readings that are more than threshold standard deviations from the mean
            loc_anomalies = group[abs(group['temperature_c'] - mean) > threshold * std]
            anomalies.append(loc_anomalies)
        
        if not anomalies:
            print("No anomalies detected.")
            return pd.DataFrame()
            
        # Combine all anomalies
        return pd.concat(anomalies)

    def save_report(self, output_path='../reports/temperature_report.txt'):
        """Generate and save a comprehensive report."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate report content
        report = []
        report.append("=" * 50)
        report.append("OFFICE TEMPERATURE MONITORING REPORT")
        report.append("=" * 50)
        report.append(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Data source: {self.data_path}")
        report.append(f"Records analyzed: {len(self.df)}")
        report.append("\n")
        
        # Overall statistics
        stats = self.get_temperature_stats()
        report.append("OVERALL TEMPERATURE STATISTICS")
        report.append("-" * 30)
        for key, value in stats.items():
            report.append(f"{key.replace('_', ' ').title()}: {value}")
        report.append("\n")
        
        # Statistics by location
        location_stats = self.get_temperature_by_location()
        report.append("TEMPERATURE BY LOCATION")
        report.append("-" * 30)
        report.append(location_stats.to_string())
        report.append("\n")
        
        # Anomalies
        anomalies = self.detect_anomalies()
        report.append("TEMPERATURE ANOMALIES")
        report.append("-" * 30)
        if len(anomalies) > 0:
            report.append(anomalies.to_string())
        else:
            report.append("No anomalies detected.")
        
        # Write report to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"Report saved to {output_path}")

# Example usage
if __name__ == "__main__":
    analyzer = TemperatureAnalyzer('../data/temperature_logs.csv')
    analyzer.plot_temperature_trends()
    analyzer.save_report()
