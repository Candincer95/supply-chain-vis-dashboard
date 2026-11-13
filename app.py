import streamlit as st
import pandas as pd
from charts import can, arkadas1, arkadas2, ml_models

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

# Top KPI Summary
col1, col2, col3 = st.columns(3)
col1.metric("Filtered Orders", f"{len(filtered_df):,}")
col2.metric("Total Sales", f"${filtered_df['Sales'].sum():,.0f}")
col3.metric("Late Delivery Rate", f"{(filtered_df['Late_delivery_risk'].mean()*100):.1f}%")

# Tabs for Each Team Member + ML Section
tab1, tab2, tab3, tab4 = st.tabs([
    "Can",
    "Member 2",
    "Member 3",
    "Machine Learning"
])

with tab1:
    can.render_charts(filtered_df)

with tab2:
    arkadas1.render_charts(filtered_df)

with tab3:
    arkadas2.render_charts(filtered_df)

with tab4:
    ml_models.render_ml(filtered_df)
