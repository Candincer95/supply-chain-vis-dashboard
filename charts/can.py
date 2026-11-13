import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def render_charts(df):
    st.header("Geospatial Distribution & Supply Chain Flow Analysis")
    
    if df.empty:
        st.warning("No data available for the selected filters. Please adjust the sidebar filters.")
        return

    # =========================================================================
    # 1. Interactive Global Bubble Map with Dynamic Metric Selector
    # =========================================================================
    st.subheader("1. Global Customer Demand, Profitability & Delivery Risk Map")
    
    map_metric = st.radio(
        "Select Map Color Metric:",
        options=["Avg Shipping Delay (Days)", "Late Delivery Rate (%)", "Net Profit ($)"],
        horizontal=True
    )

    country_coords = {
        "United States": (39.82, -98.57), "France": (46.22, 2.21), "Mexico": (23.63, -102.55),
        "Germany": (51.16, 10.45), "Brazil": (-14.23, -51.92), "United Kingdom": (55.37, -3.43),
        "Australia": (-25.27, 133.77), "China": (35.86, 104.19), "India": (20.59, 78.96),
        "Spain": (40.46, -3.74), "Italy": (41.87, 12.56), "Indonesia": (-0.78, 113.92),
        "Turkey": (38.96, 35.24), "Dominican Republic": (18.73, -70.16), "Cuba": (21.52, -77.78),
        "Colombia": (4.57, -74.29), "Argentina": (-38.41, -63.61), "El Salvador": (13.79, -88.89),
        "Honduras": (15.19, -86.24), "Nicaragua": (12.86, -85.20), "Guatemala": (15.78, -90.23),
        "Panama": (8.53, -80.78), "Netherlands": (52.13, 5.29), "Austria": (47.51, 14.55),
        "New Zealand": (-40.90, 174.88), "Philippines": (12.87, 121.77), "Nigeria": (9.08, 8.67),
        "Egypt": (26.82, 30.80), "Democratic Republic of the Congo": (-4.03, 21.75),
        "Canada": (56.13, -106.34), "Russia": (61.52, 105.31), "Japan": (36.20, 138.25),
        "South Korea": (35.90, 127.76), "South Africa": (-30.55, 22.93), "Saudi Arabia": (23.88, 45.07),
        "Poland": (51.91, 19.14), "Sweden": (60.12, 18.64), "Belgium": (50.50, 4.46),
        "Chile": (-35.67, -71.54), "Peru": (-9.18, -75.01), "Venezuela": (6.42, -66.58),
        "Thailand": (15.87, 100.99), "Vietnam": (14.05, 108.27), "Malaysia": (4.21, 101.97),
        "Pakistan": (30.37, 69.34), "Iran": (32.42, 53.68), "Iraq": (33.22, 43.67),
        "Morocco": (31.79, -7.09), "Algeria": (28.03, 1.65), "Kenya": (-0.02, 37.90),
        "Ecuador": (-1.83, -78.18), "Bolivia": (-16.29, -63.58), "Uruguay": (-32.52, -55.76),
        "Paraguay": (-23.44, -58.44), "Costa Rica": (9.74, -83.75), "Jamaica": (18.10, -77.29),
        "Haiti": (18.97, -72.28), "Portugal": (39.39, -8.22), "Switzerland": (46.81, 8.22),
        "Norway": (60.47, 8.46), "Denmark": (56.26, 9.50), "Finland": (61.92, 25.74),
        "Ireland": (53.41, -8.24), "Greece": (39.07, 21.82), "Romania": (45.94, 24.96),
        "Czech Republic": (49.81, 15.47), "Hungary": (47.16, 19.50), "Ukraine": (48.37, 31.16),
        "Israel": (31.04, 34.85), "United Arab Emirates": (23.42, 53.84), "Singapore": (1.35, 103.81),
        "Taiwan": (23.69, 120.96), "Bangladesh": (23.68, 90.35), "Sri Lanka": (7.87, 80.77),
        "Kazakhstan": (48.01, 66.92), "Uzbekistan": (41.37, 64.58), "Ghana": (7.94, -1.02),
        "Cameroon": (7.36, 12.35), "Ivory Coast": (7.54, -5.54), "Senegal": (14.49, -14.45),
        "Tanzania": (-6.36, 34.88), "Uganda": (1.37, 32.29), "Ethiopia": (9.14, 40.48),
        "Sudan": (12.86, 30.21), "Angola": (-11.20, 17.87), "Mozambique": (-18.66, 35.52),
        "Madagascar": (-18.76, 46.86), "Zambia": (-13.13, 27.84), "Zimbabwe": (-19.01, 29.15)
    }

    geo_df = df.groupby("Order Country", as_index=False).agg(
        Total_Sales=("Sales", "sum"),
        Total_Profit=("Order Profit Per Order", "sum"),
        Order_Count=("Sales", "count"),
        Avg_Delay=("Shipping Delay", "mean"),
        Late_Rate=("Late_delivery_risk", "mean")
    )
    geo_df["Lat"] = geo_df["Order Country"].map(lambda c: country_coords.get(c, (None, None))[0])
    geo_df["Lon"] = geo_df["Order Country"].map(lambda c: country_coords.get(c, (None, None))[1])
    geo_df = geo_df.dropna(subset=["Lat", "Lon"])
    geo_df["Late_Rate_Pct"] = geo_df["Late_Rate"] * 100

    metric_config = {
        "Avg Shipping Delay (Days)": ("Avg_Delay", "YlOrRd"),
        "Late Delivery Rate (%)": ("Late_Rate_Pct", "Reds"),
        "Net Profit ($)": ("Total_Profit", "Tealgrn")
    }
    color_col, color_scale = metric_config[map_metric]

    fig_map = px.scatter_map(
        geo_df,
        lat="Lat",
        lon="Lon",
        size="Total_Sales",
        color=color_col,
        hover_name="Order Country",
        hover_data={
            "Lat": False,
            "Lon": False,
            "Total_Sales": ":$,.0f",
            "Total_Profit": ":$,.0f",
            "Order_Count": ":,",
            "Avg_Delay": ":.2f",
            "Late_Rate_Pct": ":.1f"
        },
        size_max=38,
        zoom=1.15,
        center={"lat": 18.0, "lon": 10.0},
        map_style="carto-positron",
        labels={
            "Total_Sales": "Total Sales ($)",
            "Total_Profit": "Net Profit ($)",
            "Order_Count": "Total Orders",
            "Avg_Delay": "Avg Delay (Days)",
            "Late_Rate_Pct": "Late Delivery (%)"
        },
        color_continuous_scale=color_scale,
        title=f"Global Customer Orders (Bubble Size = Sales, Color = {map_metric})"
    )
    fig_map.update_layout(height=520, margin=dict(l=0, r=0, t=45, b=0))
    st.plotly_chart(fig_map, width="stretch")
    st.caption("Insight: Switch between delay days, late delivery percentage, and net profit to compare operational bottlenecks against revenue concentration.")

    st.divider()

    # =========================================================================
    # 2. Color-Coded Sankey Flow Diagram
    # =========================================================================
    st.subheader("2. Supply Chain Routing Flow: Market → Shipping Mode → Delivery Status")
    sankey_df = df.groupby(["Market", "Shipping Mode", "Delivery Status"]).size().reset_index(name="Count")

    markets = sorted(sankey_df["Market"].unique())
    modes = sorted(sankey_df["Shipping Mode"].unique())
    statuses = sorted(sankey_df["Delivery Status"].unique())

    all_nodes = markets + modes + statuses
    node_indices = {node: idx for idx, node in enumerate(all_nodes)}

    link1 = sankey_df.groupby(["Market", "Shipping Mode"])["Count"].sum().reset_index()
    link2 = sankey_df.groupby(["Shipping Mode", "Delivery Status"])["Count"].sum().reset_index()

    sources = [node_indices[m] for m in link1["Market"]] + [node_indices[sm] for sm in link2["Shipping Mode"]]
    targets = [node_indices[sm] for sm in link1["Shipping Mode"]] + [node_indices[ds] for ds in link2["Delivery Status"]]
    values = list(link1["Count"]) + list(link2["Count"])

    # Color-code links: neutral blue-gray for Market->Mode, green/red for Delivery Outcomes
    link1_colors = ["rgba(100, 149, 237, 0.30)"] * len(link1)
    link2_colors = [
        "rgba(44, 160, 44, 0.35)" if ("on time" in ds.lower() or "advance" in ds.lower()) else "rgba(214, 39, 40, 0.35)"
        for ds in link2["Delivery Status"]
    ]

    node_colors = (
        ["#1f77b4"] * len(markets) +
        ["#ff7f0e"] * len(modes) +
        ["#2ca02c" if "on time" in s.lower() or "advance" in s.lower() else "#d62728" for s in statuses]
    )

    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(
            pad=18,
            thickness=22,
            line=dict(color="black", width=0.5),
            label=all_nodes,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link1_colors + link2_colors
        )
    )])
    fig_sankey.update_layout(
        title_text="Order Volume Flow with Outcome-Colored Transitions (Red = Late/Canceled, Green = On-Time/Advance)",
        font_size=12,
        height=520
    )
    st.plotly_chart(fig_sankey, width="stretch")
    st.caption("Insight: Red ribbons immediately expose that First Class and Second Class shipments disproportionately funnel into late deliveries.")

    st.divider()

    # =========================================================================
    # 3. Interactive Top-N Sunburst Hierarchy Chart
    # =========================================================================
    st.subheader("3. Hierarchical Sales & Profitability: Market → Region → Country")
    
    top_n = st.slider("Number of Top Countries to Display in Hierarchy:", min_value=10, max_value=60, value=30, step=5)
    
    top_countries = df.groupby("Order Country")["Sales"].sum().nlargest(top_n).index
    sunburst_df = df[df["Order Country"].isin(top_countries)].groupby(
        ["Market", "Order Region", "Order Country"], as_index=False
    ).agg(
        Total_Sales=("Sales", "sum"),
        Total_Profit=("Order Profit Per Order", "sum")
    )

    fig_sunburst = px.sunburst(
        sunburst_df,
        path=["Market", "Order Region", "Order Country"],
        values="Total_Sales",
        color="Total_Profit",
        color_continuous_scale="Blues",
        labels={
            "Total_Sales": "Total Sales ($)",
            "Total_Profit": "Net Profit ($)"
        },
        title=f"Interactive Drill-Down of Sales & Net Profit (Top {top_n} Countries — Click Slices to Zoom)"
    )
    fig_sunburst.update_layout(height=580, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_sunburst, width="stretch")
    st.caption("Insight: Click any market or region slice to zoom into country-level sales share and net profitability.")