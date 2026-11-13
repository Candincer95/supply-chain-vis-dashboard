import streamlit as st
import pandas as pd
from charts.member1_geo import render_member1_charts
from charts.member2_product import render_member2_charts
from charts.member3_ml import render_member3_charts

st.set_page_config(page_title="Supply Chain Analytics Dashboard", layout="wide")
st.title("Global Supply Chain Analytics & Delivery Risk Prediction Dashboard")

@st.cache_data
def load_data():
    return pd.read_csv("cleaned_supply_chain.csv")

df = load_data()

# Sidebar Interactive Filters
st.sidebar.header("Dashboard Filters")

selected_markets = st.sidebar.multiselect(
    "Select Market(s)",
    options=sorted(df["Market"].unique()),
    default=sorted(df["Market"].unique())
)

sales_range = st.sidebar.slider(
    "Sales Amount Range ($)",
    float(df["Sales"].min()),
    float(df["Sales"].max()),
    (float(df["Sales"].min()), float(df["Sales"].max()))
)

only_late = st.sidebar.checkbox("Show Only Late Deliveries", value=False)

filtered_df = df[
    (df["Market"].isin(selected_markets)) &
    (df["Sales"].between(sales_range[0], sales_range[1]))
]
if only_late:
    filtered_df = filtered_df[filtered_df["Late_delivery_risk"] == 1]

# Top KPI Cards
col1, col2, col3 = st.columns(3)
col1.metric("Filtered Orders", f"{len(filtered_df):,}")
col2.metric("Total Sales", f"${filtered_df['Sales'].sum():,.0f}")
col3.metric("Late Delivery Rate", f"{(filtered_df['Late_delivery_risk'].mean()*100):.1f}%")

# Modular Tabs for Each Team Member
tab1, tab2, tab3 = st.tabs([
    "Logistics & Geo Flow (Member 1)",
    "Product & Profitability (Member 2)",
    "Trends & ML Predictions (Member 3)"
])

with tab1:
    render_member1_charts(filtered_df)

with tab2:
    render_member2_charts(filtered_df)

with tab3:
    render_member3_charts(filtered_df)
