import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def _sync_treemap_metric():
    # Linked controls: changing the timeline metric also recolours the treemap
    st.session_state["umran_treemap_metric"] = st.session_state["umran_cs_metric"]


def render_charts(df):
    st.header("Temporal Performance, Product Portfolio & Pricing Behaviour")

    if df.empty:
        st.warning("No data available for the selected filters. Please adjust the sidebar filters.")
        return

    # =========================================================================
    # 1. Connected Scatterplot (Timeline) - acts as the brushing view
    # =========================================================================
    st.subheader("1. Performance Trajectory Over Time: Connected Scatterplot")

    col_a, col_b, col_c = st.columns([2, 2, 1.4])
    category_options = ["All Categories"] + list(
        df.groupby("Category Name")["Sales"].sum().sort_values(ascending=False).index
    )
    selected_category = col_a.selectbox(
        "Product category:", options=category_options, key="umran_cs_category"
    )
    y_metric = col_b.radio(
        "Y-axis metric:",
        options=["Late Delivery Rate (%)", "Profit Margin (%)"],
        horizontal=True,
        key="umran_cs_metric",
        on_change=_sync_treemap_metric
    )
    granularity = col_c.radio(
        "Time step:",
        options=["Quarterly", "Monthly"],
        horizontal=True,
        key="umran_cs_granularity"
    )
    hide_low_volume = st.checkbox(
        "Hide low-volume periods (fewer than half the typical number of orders)",
        value=True,
        key="umran_cs_hide_low"
    )

    # Category filter (shared by all three charts)
    all_df = df.copy()
    if granularity == "Quarterly":
        all_df["Period"] = pd.PeriodIndex(all_df["Order Month"], freq="M").asfreq("Q").astype(str)
    else:
        all_df["Period"] = all_df["Order Month"].astype(str)
    cat_df = all_df if selected_category == "All Categories" else all_df[all_df["Category Name"] == selected_category]

    periods = cat_df.groupby("Period", as_index=False).agg(
        Total_Sales=("Sales", "sum"),
        Total_Profit=("Order Profit Per Order", "sum"),
        Late_Rate=("Late_delivery_risk", "mean"),
        Order_Count=("Sales", "count")
    ).sort_values("Period")
    # Always ignore periods with very few orders (statistically noisy)
    periods = periods[periods["Order_Count"] >= 10]
    hidden_periods = []
    if hide_low_volume and len(periods) > 0:
        typical = periods["Order_Count"].median()
        low_mask = periods["Order_Count"] < 0.5 * typical
        hidden_periods = list(periods.loc[low_mask, "Period"])
        periods = periods[~low_mask]
    periods = periods.reset_index(drop=True)

    selected_periods = []
    if len(periods) < 2:
        st.info("Not enough data over time for this selection. Try another category or widen the filters.")
    else:
        periods["Late_Rate_Pct"] = periods["Late_Rate"] * 100
        periods["Profit_Margin"] = periods["Total_Profit"] / periods["Total_Sales"] * 100
        y_col = "Late_Rate_Pct" if y_metric == "Late Delivery Rate (%)" else "Profit_Margin"
        period_idx = list(range(len(periods)))
        period_word = "Quarterly" if granularity == "Quarterly" else "Monthly"

        fig_cs = go.Figure()
        fig_cs.add_trace(go.Scatter(
            x=periods["Total_Sales"],
            y=periods[y_col],
            mode="lines+markers",
            line=dict(color="rgba(120, 120, 120, 0.45)", width=1.5),
            marker=dict(
                size=12,
                color=period_idx,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(
                    title="Time",
                    tickvals=[0, len(periods) - 1],
                    ticktext=[periods["Period"].iloc[0], periods["Period"].iloc[-1]],
                    outlinewidth=0
                ),
                line=dict(color="white", width=1)
            ),
            customdata=periods[["Period", "Order_Count"]],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Sales: $%{x:,.0f}<br>"
                f"{y_metric}: " + "%{y:.1f}<br>"
                "Order Items: %{customdata[1]:,}<extra></extra>"
            ),
            showlegend=False
        ))

        for idx, label in [(0, "Start"), (len(periods) - 1, "End")]:
            fig_cs.add_annotation(
                x=periods["Total_Sales"].iloc[idx],
                y=periods[y_col].iloc[idx],
                text=f"{label}: {periods['Period'].iloc[idx]}",
                showarrow=True, arrowhead=2, ax=30, ay=-30,
                font=dict(size=11)
            )

        fig_cs.update_layout(
            title=f"{selected_category}: {period_word} Sales vs {y_metric} (Select points to filter the charts below)",
            xaxis_title=f"{period_word} Sales ($)",
            yaxis_title=y_metric,
            height=520,
            margin=dict(l=10, r=10, t=60, b=10),
            dragmode="select"
        )

        event = st.plotly_chart(
            fig_cs,
            width="stretch",
            on_select="rerun",
            selection_mode=("points", "box", "lasso"),
            key="umran_cs_chart"
        )

        # Read the brushed points (works whether the event is returned as a dict or an object)
        point_indices = []
        try:
            selection = event["selection"] if isinstance(event, dict) else event.selection
            point_indices = list(selection.get("point_indices", [])) if isinstance(selection, dict) \
                else list(getattr(selection, "point_indices", []))
            if not point_indices:
                pts = selection["points"] if isinstance(selection, dict) else selection.points
                point_indices = [p.get("point_index", p.get("point_number")) for p in pts]
        except Exception:
            point_indices = []
        selected_periods = [
            periods["Period"].iloc[i] for i in point_indices
            if i is not None and 0 <= i < len(periods)
        ]

        if hidden_periods:
            st.caption(f"Not shown because they have too few orders: {', '.join(hidden_periods)}. Uncheck the box above to include them.")

        peak_row = periods.loc[periods[y_col].idxmax()]
        best_sales_row = periods.loc[periods["Total_Sales"].idxmax()]
        st.caption(
            f"Insight: For {selected_category.lower()}, the highest {y_metric.lower()} occurred in {peak_row['Period']} "
            f"({peak_row[y_col]:.1f}), while the strongest sales period was {best_sales_row['Period']} "
            f"(${best_sales_row['Total_Sales']:,.0f}). Following the path reveals whether higher sales volume coincides "
            f"with operational stress (more late deliveries) or weaker margins."
        )

    # -------------------------------------------------------------------------
    # Build the linked datasets
    #   - Treemap: linked to the selected time periods (all categories, so the
    #     portfolio stays comparable)
    #   - Parallel coordinates: linked to the selected periods AND the category
    # -------------------------------------------------------------------------
    period_df = all_df
    linked_df = cat_df
    if selected_periods:
        period_df = all_df[all_df["Period"].isin(selected_periods)]
        linked_df = cat_df[cat_df["Period"].isin(selected_periods)]
    # Compact one-line indicator of the active linked filters
    if selected_periods or selected_category != "All Categories":
        parts = []
        if selected_periods:
            parts.append("Periods " + " ".join(f":blue-background[{p}]" for p in sorted(selected_periods)))
        if selected_category != "All Categories":
            parts.append(f"Category :blue-background[{selected_category}] *(parallel coordinates)*")
        hint = " · *double-click the timeline to reset*" if selected_periods else ""
        st.markdown("🔗 **Linked filter:** " + " &nbsp;·&nbsp; ".join(parts) + hint)

    if period_df.empty or linked_df.empty:
        st.warning("No orders match the current linked selection.")
        return

    st.divider()

    # =========================================================================
    # 2. Product Hierarchy Treemap (Department -> Category -> Product)
    # =========================================================================
    st.subheader("2. Product Portfolio Treemap: Department → Category → Product")

    if "umran_treemap_metric" not in st.session_state:
        st.session_state["umran_treemap_metric"] = y_metric  # start in sync with the timeline
    treemap_metric = st.selectbox(
        "Color treemap tiles by:",
        options=["Profit Margin (%)", "Late Delivery Rate (%)", "Avg Discount Rate (%)"],
        key="umran_treemap_metric"
    )

    # Colorblind-safe diverging ColorBrewer scale (red = worse, blue = better than average)
    treemap_config = {
        "Profit Margin (%)": ("margin", "RdBu", "higher margin", "lower margin"),
        "Late Delivery Rate (%)": ("late", "RdBu_r", "fewer late deliveries", "more late deliveries"),
        "Avg Discount Rate (%)": ("disc", "RdBu_r", "lower discounts", "higher discounts")
    }
    metric_key, color_scale, good_text, bad_text = treemap_config[treemap_metric]

    def node_stats(frame, keys):
        g = frame.groupby(keys).agg(
            Sales=("Sales", "sum"),
            Profit=("Order Profit Per Order", "sum"),
            Late=("Late_delivery_risk", "mean"),
            Disc=("Order Item Discount Rate", "mean"),
            Count=("Sales", "count")
        ).reset_index()
        g = g[g["Sales"] > 0]
        g["margin"] = g["Profit"] / g["Sales"] * 100
        g["late"] = g["Late"] * 100
        g["disc"] = g["Disc"] * 100
        return g

    ids, labels, parents, values, colors, custom = [], [], [], [], [], []
    root_sales = period_df["Sales"].sum()
    root_stats = {
        "margin": period_df["Order Profit Per Order"].sum() / root_sales * 100 if root_sales else 0.0,
        "late": period_df["Late_delivery_risk"].mean() * 100,
        "disc": period_df["Order Item Discount Rate"].mean() * 100
    }
    center = root_stats[metric_key]
    ids.append("All Products"); labels.append("All Products"); parents.append(""); values.append(0)
    colors.append(center)
    custom.append([center, root_sales, period_df["Order Profit Per Order"].sum(), len(period_df),
                   root_stats["margin"], root_stats["late"], root_stats["disc"]])

    level_keys = [
        ["Department Name"],
        ["Department Name", "Category Name"],
        ["Department Name", "Category Name", "Product Name"]
    ]
    for depth, keys in enumerate(level_keys):
        stats = node_stats(period_df, keys)
        is_leaf = depth == len(level_keys) - 1
        for _, r in stats.iterrows():
            path = [str(r[k]) for k in keys]
            ids.append("/".join(path))
            labels.append(path[-1])
            parents.append("/".join(path[:-1]) if len(path) > 1 else "All Products")
            values.append(float(r["Sales"]) if is_leaf else 0)
            colors.append(float(r[metric_key]))
            custom.append([float(r[metric_key]), float(r["Sales"]), float(r["Profit"]), int(r["Count"]),
                           float(r["margin"]), float(r["late"]), float(r["disc"])])

    # Colour range from the category-level spread around the average
    cat_stats = node_stats(period_df, ["Category Name"])
    spread = float((cat_stats[metric_key] - center).abs().quantile(0.9)) if len(cat_stats) > 1 else 5.0
    spread = max(spread, 2.0)
    # Keep the colour range inside 0-100 for rates (late delivery, discount)
    cmin, cmax = center - spread, center + spread
    if metric_key in ("late", "disc"):
        cmin, cmax = max(0.0, cmin), min(100.0, cmax)
        if cmax - cmin < 1e-9:
            cmin, cmax = max(0.0, center - 1), min(100.0, center + 1)
    no_variation = len(cat_stats) > 0 and float(cat_stats[metric_key].max() - cat_stats[metric_key].min()) < 1e-9

    fig_tree = go.Figure(go.Treemap(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="remainder",
        maxdepth=3,
        customdata=custom,
        marker=dict(
            colors=colors,
            colorscale=color_scale,
            cmid=center,
            cmin=cmin,
            cmax=cmax,
            showscale=not no_variation,
            colorbar=dict(title=treemap_metric, outlinewidth=0),
            line=dict(color="white", width=1)
        ),
        texttemplate="<b>%{label}</b><br>%{customdata[0]:.1f}%<br>$%{customdata[1]:,.0f}",
        textfont=dict(size=13),
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Sales: $%{customdata[1]:,.0f}<br>"
            "Net Profit: $%{customdata[2]:,.0f}<br>"
            "Order Items: %{customdata[3]:,}<br>"
            "Profit Margin: %{customdata[4]:.1f}%<br>"
            "Late Delivery: %{customdata[5]:.1f}%<br>"
            "Avg Discount: %{customdata[6]:.1f}%<extra></extra>"
        )
    ))
    fig_tree.update_layout(
        title=f"Tile Size = Sales, Color = {treemap_metric} vs. Average ({center:.1f}%) — Click a category to see its products",
        height=600,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    st.plotly_chart(fig_tree, width="stretch", key="umran_treemap_chart")
    if no_variation:
        st.caption(
            f"Every category has the same {treemap_metric.lower()} ({center:.1f}%) for the current filters "
            f"(e.g. 'Show Only Late Deliveries' makes every late rate 100%), so the colours show no difference. "
            f"Choose another colour metric to compare categories."
        )

    if len(cat_stats) > 0:
        cat_idx = cat_stats.set_index("Category Name")
        top_cat = cat_idx["Sales"].idxmax()
        worst_cat = cat_idx["margin"].idxmin()
        if top_cat == worst_cat:
            margin_text = (
                f"'{top_cat}' generates the largest sales volume but also has the weakest profit margin "
                f"({cat_idx.loc[top_cat, 'margin']:.1f}%)."
            )
        else:
            margin_text = (
                f"'{top_cat}' generates the largest sales volume with a {cat_idx.loc[top_cat, 'margin']:.1f}% "
                f"profit margin, while '{worst_cat}' has the weakest margin ({cat_idx.loc[worst_cat, 'margin']:.1f}%)."
            )
        st.caption(
            f"Insight: {margin_text} Colors use a colorblind-safe diverging scale centred on the overall average "
            f"({center:.1f}%): blue tiles mean {good_text}, red tiles mean {bad_text} than average."
        )

    st.divider()

    # =========================================================================
    # 3. Parallel Coordinates Plot (Pricing -> Profit -> Logistics)
    # =========================================================================
    st.subheader("3. Multivariate Order Profile: Parallel Coordinates")

    sample_size = st.slider(
        "Number of orders to display (random sample to reduce clutter):",
        min_value=200, max_value=3000, value=1000, step=100,
        key="umran_pcp_sample"
    )

    pcp_cols = [
        "Order Item Quantity",
        "Order Item Discount Rate",
        "Sales",
        "Order Profit Per Order",
        "Days for shipping (real)",
        "Shipping Delay"
    ]
    pcp_df = linked_df[pcp_cols].dropna()
    pcp_df = pcp_df.sample(n=min(sample_size, len(pcp_df)), random_state=42)

    fig_pcp = px.parallel_coordinates(
        pcp_df,
        dimensions=pcp_cols,
        color="Order Profit Per Order",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        labels={
            "Order Item Quantity": "Quantity",
            "Order Item Discount Rate": "Discount Rate",
            "Sales": "Sales ($)",
            "Order Profit Per Order": "Profit ($)",
            "Days for shipping (real)": "Shipping Days",
            "Shipping Delay": "Delay (Days)"
        },
        title=f"{len(pcp_df):,} Sampled Orders (Line Color = Profit: Red = Loss, Blue = Profit)"
    )
    fig_pcp.update_layout(height=520, margin=dict(l=60, r=60, t=80, b=30), coloraxis_colorbar=dict(outlinewidth=0))
    st.plotly_chart(fig_pcp, width="stretch", key="umran_pcp_chart")

    loss_share = (linked_df["Order Profit Per Order"] < 0).mean() * 100
    disc_corr = linked_df["Order Item Discount Rate"].corr(linked_df["Order Profit Per Order"])
    disc_corr_text = f"{disc_corr:.2f}" if pd.notna(disc_corr) else "n/a"
    st.caption(
        f"Insight: {loss_share:.1f}% of the selected orders are sold at a loss. The correlation between discount rate "
        f"and profit is {disc_corr_text}. Drag along any axis to brush (filter) a value range, and drag axis titles "
        f"to reorder dimensions and compare different variable pairs."
    )
