import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Malaysia Water Data Dashboard",
    page_icon="💧",
    layout="wide"
)

# Title and description
st.title("Malaysia Water Resources Analysis Dashboard")
st.markdown("""
This interactive dashboard visualizes water production, consumption, and access trends 
across Malaysian states over time. Use the sidebar filters to customize your view!
""")

# Load the preprocessed data
@st.cache_data
def load_data():
    df = pd.read_csv("data/preprocessed_water_data.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

# Load data
df = load_data()

# Sidebar filters
st.sidebar.header("Filters")

# Time range filter
min_year = df['date'].dt.year.min()
max_year = df['date'].dt.year.max()
year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=int(min_year),
    max_value=int(max_year),
    value=(2003, 2020)
)

# State selection filter
all_states = sorted(df['state'].unique())
selected_states = st.sidebar.multiselect(
    "Select States",
    options=all_states,
    default=["Malaysia"]
)

# Apply filters
filtered_df = df[
    (df['date'].dt.year >= year_range[0]) & 
    (df['date'].dt.year <= year_range[1]) &
    (df['state'].isin(selected_states))
]

# Main dashboard content
cols = st.columns([1, 1])

with cols[0]:
    st.subheader("Water Production vs Consumption Trends")
    
    # Create a plotly figure for production and consumption trends
    if not filtered_df.empty:
        # For production vs consumption, group by date and sum across selected states
        grouped_data = filtered_df.groupby('date')[['production_mld', 'consump_total_mld']].sum().reset_index()
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        
        # Add traces
        fig.add_trace(
            go.Scatter(
                x=grouped_data['date'], 
                y=grouped_data['production_mld'],
                name="Production (MLD)",
                line=dict(color='blue', width=3)
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=grouped_data['date'], 
                y=grouped_data['consump_total_mld'],
                name="Consumption (MLD)",
                line=dict(color='red', width=3)
            )
        )
        
        # Set x-axis title
        fig.update_xaxes(title_text="Year")
        
        # Set y-axes titles
        fig.update_yaxes(title_text="Million Liters per Day (MLD)")
        
        # Update layout
        fig.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        
        # Show plot
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available for the selected filters.")

with cols[1]:
    st.subheader("Water Consumption Breakdown")
    
    # Create a plotly figure for domestic vs non-domestic consumption
    if not filtered_df.empty:
        # For consumption breakdown, group by date and sum across selected states
        grouped_data = filtered_df.groupby('date')[['consump_domestic_mld', 'consump_nondomestic_mld']].sum().reset_index()
        
        # Create figure with secondary y-axis
        fig = go.Figure()
        
        # Add traces
        fig.add_trace(
            go.Scatter(
                x=grouped_data['date'], 
                y=grouped_data['consump_domestic_mld'],
                name="Domestic Consumption",
                line=dict(color='green', width=3)
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=grouped_data['date'], 
                y=grouped_data['consump_nondomestic_mld'],
                name="Non-Domestic Consumption",
                line=dict(color='orange', width=3)
            )
        )
        
        # Set x-axis title
        fig.update_xaxes(title_text="Year")
        
        # Set y-axes titles
        fig.update_yaxes(title_text="Million Liters per Day (MLD)")
        
        # Update layout
        fig.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        
        # Show plot
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available for the selected filters.")

# Second row with interactive controls
st.subheader("Water Access Analysis")

# Create access analysis with slider for year
selected_year = st.slider(
    "Select Year for Access Analysis", 
    min_value=int(min_year), 
    max_value=int(max_year),
    value=2020
)

# Filter by selected year for the access analysis
year_data = df[df['date'].dt.year == selected_year]

if not year_data.empty and not year_data['access_overall_pct'].isnull().all():
    # Sort states by overall access percentage (excluding Malaysia for better visualization)
    state_access = year_data[year_data['state'] != 'Malaysia'].sort_values('access_overall_pct', ascending=False)
    
    # Create data subsets for access visualization
    cols2 = st.columns([1, 1])
    
    with cols2[0]:
        # Create a bar chart for overall water access by state
        fig = px.bar(
            state_access,
            x='state',
            y='access_overall_pct',
            title=f'Overall Water Access by State ({selected_year})',
            labels={'access_overall_pct': 'Access %', 'state': 'State'},
            color='access_overall_pct',
            color_continuous_scale='Blues',
            height=500
        )
        fig.update_layout(xaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig, use_container_width=True)
        
    with cols2[1]:
        # Create a comparative bar chart for urban vs rural water access
        if 'access_rural_pct' in state_access.columns and 'access_urban_pct' in state_access.columns:
            # Create a DataFrame for the grouped bar chart
            data_for_chart = []
            for _, row in state_access.iterrows():
                if pd.notna(row['access_rural_pct']) and pd.notna(row['access_urban_pct']):
                    data_for_chart.append({'State': row['state'], 'Type': 'Rural', 'Access %': row['access_rural_pct']})
                    data_for_chart.append({'State': row['state'], 'Type': 'Urban', 'Access %': row['access_urban_pct']})
            
            chart_df = pd.DataFrame(data_for_chart)
            
            if not chart_df.empty:
                fig = px.bar(
                    chart_df,
                    x='State',
                    y='Access %',
                    color='Type',
                    barmode='group',
                    title=f'Urban vs. Rural Water Access by State ({selected_year})',
                    height=500,
                    color_discrete_map={'Urban': 'darkblue', 'Rural': 'skyblue'}
                )
                fig.update_layout(xaxis={'categoryorder':'total descending'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No urban/rural access data available for the selected year.")
        else:
            st.warning("Urban and rural access data not available.")
else:
    st.warning(f"No water access data available for {selected_year}.")

# NEW SECTION - Water Stress Indicator with interactive threshold
st.subheader("Water Stress Indicator Analysis")

# Create a slider for the Water Stress Threshold
stress_threshold = st.slider(
    "Select Water Stress Threshold (Consumption/Production Ratio)", 
    min_value=0.5, 
    max_value=1.0, 
    value=0.7,
    step=0.05,
    help="Values closer to 1.0 indicate higher water stress. Above 1.0 means consumption exceeds production."
)

# Select year for stress analysis using a slider
selected_year_stress = st.slider(
    "Select Year for Water Stress Analysis", 
    min_value=int(min_year), 
    max_value=int(max_year),
    value=2015
)

# Filter by selected year for stress analysis
year_data_stress = df[df['date'].dt.year == selected_year_stress]

if (not year_data_stress.empty and 
    not year_data_stress['stress_indicator_consump_prod'].isnull().all()):
    
    # Sort states by stress indicator
    state_stress = year_data_stress[year_data_stress['state'] != 'Malaysia'].copy()
    
    # Add a column to categorize stress levels based on the threshold
    state_stress['stress_category'] = np.where(
        state_stress['stress_indicator_consump_prod'] >= stress_threshold, 
        'High Stress', 
        'Low Stress'
    )
    
    # Create a bar chart for stress indicator
    fig = px.bar(
        state_stress.sort_values('stress_indicator_consump_prod', ascending=False),
        x='state',
        y='stress_indicator_consump_prod',
        title=f'Water Stress Indicator by State ({selected_year_stress})',
        labels={'stress_indicator_consump_prod': 'Consumption/Production Ratio', 'state': 'State'},
        color='stress_category',
        color_discrete_map={'High Stress': 'red', 'Low Stress': 'green'},
        height=500
    )
    
    # Add a horizontal line at the threshold value
    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=len(state_stress),
        y0=stress_threshold,
        y1=stress_threshold,
        line=dict(color="black", width=2, dash="dash")
    )
    
    # Add annotation for threshold line
    fig.add_annotation(
        x=len(state_stress)-1,
        y=stress_threshold * 1.05,
        text=f"Threshold: {stress_threshold}",
        showarrow=False,
        font=dict(size=12, color="black")
    )
    
    fig.update_layout(xaxis={'categoryorder':'total descending'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Display counts for high/low stress
    high_stress_count = sum(state_stress['stress_indicator_consump_prod'] >= stress_threshold)
    low_stress_count = sum(state_stress['stress_indicator_consump_prod'] < stress_threshold)
    
    # Create columns for metrics
    metric_cols = st.columns(2)
    with metric_cols[0]:
        st.metric(
            "States with High Water Stress",
            high_stress_count,
            f"{high_stress_count/len(state_stress)*100:.1f}% of states"
        )
    with metric_cols[1]:
        st.metric(
            "States with Low Water Stress",
            low_stress_count,
            f"{low_stress_count/len(state_stress)*100:.1f}% of states"
        )
else:
    st.warning(f"No water stress data available for {selected_year_stress}.")

# Third row - Per Capita Consumption
st.subheader("Per Capita Water Consumption Analysis")

cols3 = st.columns([2, 1])

with cols3[0]:
    # Select year for per capita analysis using a slider
    selected_year_capita = st.slider(
        "Select Year for Per Capita Analysis", 
        min_value=int(min_year), 
        max_value=int(max_year),
        value=2015
    )
    
    # Filter by selected year for per capita analysis
    year_data_capita = df[df['date'].dt.year == selected_year_capita]
    
    if (not year_data_capita.empty and 
        not year_data_capita['consump_per_capita_lpd'].isnull().all()):
        
        # Sort states by per capita consumption
        state_capita = year_data_capita[year_data_capita['state'] != 'Malaysia'].sort_values('consump_per_capita_lpd', ascending=False)
        
        # Create a bar chart for per capita consumption
        fig = px.bar(
            state_capita,
            x='state',
            y='consump_per_capita_lpd',
            title=f'Per Capita Water Consumption by State ({selected_year_capita})',
            labels={'consump_per_capita_lpd': 'Liters per Day per Person', 'state': 'State'},
            color='consump_per_capita_lpd',
            color_continuous_scale='YlOrRd',
            height=500
        )
        fig.update_layout(xaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No per capita consumption data available for {selected_year_capita}.")

with cols3[1]:
    st.markdown("### Key Indicators")
    
    # Filter for Malaysia data for the selected year range
    malaysia_data = df[
        (df['date'].dt.year >= year_range[0]) & 
        (df['date'].dt.year <= year_range[1]) &
        (df['state'] == 'Malaysia')
    ]
    
    # Display key metrics
    if not malaysia_data.empty:
        latest_year = malaysia_data['date'].dt.year.max()
        latest_data = malaysia_data[malaysia_data['date'].dt.year == latest_year].iloc[0]
        earliest_data = malaysia_data[malaysia_data['date'].dt.year == malaysia_data['date'].dt.year.min()].iloc[0]
        
        # Calculate percentage changes
        try:
            pop_change = ((latest_data['population'] - earliest_data['population']) / earliest_data['population']) * 100
            production_change = ((latest_data['production_mld'] - earliest_data['production_mld']) / earliest_data['production_mld']) * 100
            consumption_change = ((latest_data['consump_total_mld'] - earliest_data['consump_total_mld']) / earliest_data['consump_total_mld']) * 100
            
            # Display metrics
            st.metric("Population Growth", f"{pop_change:.1f}%", "")
            st.metric("Water Production Growth", f"{production_change:.1f}%", "")
            st.metric("Water Consumption Growth", f"{consumption_change:.1f}%", "")
            
            # Access metrics if available
            if pd.notna(latest_data['access_overall_pct']):
                access_change = latest_data['access_overall_pct'] - earliest_data['access_overall_pct']
                st.metric("Water Access Change", f"{access_change:.1f}%", "")
        except:
            st.warning("Could not calculate some metrics due to missing data")
    else:
        st.warning("No data available for Malaysia in the selected year range")

# Footer
st.markdown("---")
st.markdown("Data source: Malaysian water resources dataset")
st.markdown("Dashboard created with Streamlit") 