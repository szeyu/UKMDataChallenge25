# Malaysia Water Resources Interactive Dashboard

This interactive Streamlit dashboard visualizes water production, consumption, and access trends across Malaysian states over time.

## Features

- **Interactive Time Range Selection**: Use the slider to select the year range for analysis
- **Multi-State Selection**: Choose one or multiple states to compare data
- **Dynamic Visualizations**: Charts update automatically based on your selections
- **Per Capita Analysis**: Explore water consumption per person with year-by-year slider
- **Urban vs Rural Comparison**: Analyze the differences in water access between urban and rural areas
- **Water Stress Analysis**: Adjust the stress threshold slider to identify states with high water stress

## Data Overview

The dashboard uses these key datasets:
- Population data by state
- Water production data
- Water consumption (domestic and non-domestic)
- Water access percentages (overall, urban, rural)

## Getting Started

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

3. Open your browser and navigate to `http://localhost:8501` to view the dashboard.

## Dashboard Sections

1. **Top Row**: Water production vs consumption trends and consumption breakdown
2. **Middle Row**: Water access analysis with year slider and water stress indicator with threshold slider
3. **Bottom Row**: Per capita water consumption with year slider and key metrics

## Interactive Elements

- **Year Range Slider**: Filter the entire dashboard by a time period
- **State Selection**: Multi-select dropdown for comparing different states
- **Year Sliders**: Individual year selectors for access, stress, and per capita analyses
- **Stress Threshold Slider**: Dynamically adjust what constitutes "high stress" for water resources

## Data Source

Data sourced from Malaysian water resources datasets, preprocessed for dashboard visualization.