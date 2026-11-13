import streamlit as st
import pandas as pd
from charts import can, umran, berkay, ml_models

# Page Configuration
st.set_page_config(
    page_title="Global Supply Chain Analytics Dashboard",
    layout="wide"
)

st.title("Global Supply Chain Analytics & Delivery Risk Dashboard")
st.markdown("Interactive exploration of global orders, logistics performance, profitability and delivery risk.")

# Load Preprocessed Dataset with Caching
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_supply_chain.csv")

df = load_data()

# Sidebar Interactive Filters
st.sidebar.header("Dashboard Filters")

selected_markets = st.sidebar.multiselect(
    "1. Select Market(s)",
    options=sorted(df["Market"].unique()),
    default=sorted(df["Market"].unique())
)

sales_min = float(df["Sales"].min())
sales_max = float(df["Sales"].max())
sales_range = st.sidebar.slider(
    "2. Sales Amount Range ($)",
    min_value=sales_min,
    max_value=sales_max,
    value=(sales_min, sales_max)
)

only_late = st.sidebar.checkbox(
    "3. Show Only Late Deliveries",
    value=False
)

# Apply Filters to the Dataset
filtered_df = df[
    (df["Market"].isin(selected_markets)) &
    (df["Sales"].between(sales_range[0], sales_range[1]))
]

if only_late:
    filtered_df = filtered_df[filtered_df["Late_delivery_risk"] == 1]

st.sidebar.markdown("---")
st.sidebar.info(f"Showing **{len(filtered_df):,}** of **{len(df):,}** total orders.")

# Top KPI Summary Cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Filtered Orders", f"{len(filtered_df):,}")
col2.metric("Total Sales", f"${filtered_df['Sales'].sum():,.0f}")
col3.metric("Total Profit", f"${filtered_df['Order Profit Per Order'].sum():,.0f}")

late_rate = (filtered_df["Late_delivery_risk"].mean() * 100) if len(filtered_df) >0 else 0.0
col4.metric("Late Delivery Rate", f"{late_rate:.1f}%")

st.markdown("---")

# Modular Tabs for Team Members & Machine Learning
tab1, tab2, tab3, tab4 = st.tabs([
    "Can",
    "Umran",
    "Berkay",
    "Machine Learning Models"
])

with tab1:
    can.render_charts(filtered_df)

with tab2:
    umran.render_charts(filtered_df)

with tab3:
    berkay.render_charts(filtered_df)

with tab4:
    ml_models.render_ml(filtered_df)