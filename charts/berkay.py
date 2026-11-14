import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def monthly_sales_tables(df):
    data = df.copy()
    data["Month"] = pd.to_datetime(data["Order Month"]).dt.to_period("M").dt.to_timestamp()
    data["Department Name"] = data["Department Name"].str.strip()
    grouped = data.groupby(["Department Name", "Month"]).agg(
        Sales=("Sales", "sum"), Records=("Sales", "size")
    ).reset_index()
    sales = grouped.pivot(index="Department Name", columns="Month", values="Sales")
    counts = grouped.pivot(index="Department Name", columns="Month", values="Records")
    order = sales.sum(axis=1).sort_values(ascending=False).index
    return sales.reindex(order), counts.reindex(order)


def render_portfolio(df):
    st.subheader("1. Monthly sales by product department")
    sales, counts = monthly_sales_tables(df)
    years = sorted(sales.columns.year.unique().tolist())
    complete_candidates = [y for y in years if (sales.columns.year == y).sum() > 1]
    default_year = max(complete_candidates) if complete_candidates else max(years)
    year = st.selectbox(
        "Year", years, index=years.index(default_year), key="berkay_heatmap_year"
    )
    observed = sales.columns[sales.columns.year == year]
    months = pd.date_range(observed.min(), observed.max(), freq="MS")
    shown = sales.reindex(columns=months)
    shown_counts = counts.reindex(columns=months)
    values = shown.to_numpy() / 1000
    maximum = max(1.0, float(np.nanmax(sales.to_numpy())) / 1000)
    details = np.stack([shown.to_numpy(), shown_counts.to_numpy()], axis=-1)
    fig = go.Figure(go.Heatmap(
        x=months.strftime("%b").tolist(),
        y=shown.index.tolist(), z=values, customdata=details,
        colorscale="YlGnBu", zmin=0, zmax=maximum,
        colorbar=dict(title="Sales (thousands)"),
        texttemplate="%{z:.1f}", textfont=dict(size=12),
        xgap=3, ygap=3, hoverongaps=False,
        hovertemplate=(
            "%{y}<br>%{x} " + str(year) +
            "<br>Sales: %{customdata[0]:,.2f}"
            "<br>Order-item rows: %{customdata[1]:,.0f}<extra></extra>"
        )
    ))
    fig.update_layout(
        height=max(400, 38 * len(shown) + 130), template="plotly_white",
        xaxis_title="Order month", yaxis_title="Product department",
        yaxis=dict(autorange="reversed"), margin=dict(l=150, r=30, t=30, b=70)
    )
    st.plotly_chart(fig, width="stretch", key="berkay_portfolio")
    st.caption(
        "Numbers show sales in thousands. Darker cells mean higher sales. "
        "Compare down a column to rank departments in a month, "
        "or across a row to follow a department over time. Hover for exact amounts."
    )
    st.caption(
        "Blank cells have no observations in the filtered sample, not confirmed zero demand. "
        "The color scale stays the same across years for the current sidebar filters. "
        "Changes describe the sampled data; they do not by themselves establish "
        "business growth, product launches or declining demand."
    )


def shipping_summary(df):
    active = df[df["Delivery Status"] != "Shipping canceled"]
    summary = active.groupby("Shipping Mode").agg(
        Planned=("Days for shipment (scheduled)", "mean"),
        Actual=("Days for shipping (real)", "mean"),
        Records=("Sales", "size"),
        Late_rate=("Late_delivery_risk", "mean")
    )
    order = ["Same Day", "First Class", "Second Class", "Standard Class"]
    summary = summary.reindex([mode for mode in order if mode in summary.index])
    summary["Gap"] = summary["Actual"] - summary["Planned"]
    return summary, len(df) - len(active)


def render_shipping(df):
    st.subheader("2. Scheduled vs. actual shipping time")
    summary, excluded = shipping_summary(df)
    if summary.empty:
        st.info("No non-canceled shipments remain in the selected data.")
        return
    x, y = [], []
    for mode, row in summary.iterrows():
        x.extend([row["Planned"], row["Actual"], None])
        y.extend([mode, mode, None])
    details = np.column_stack([
        summary["Records"], summary["Gap"], 100 * summary["Late_rate"]
    ])
    fig = go.Figure(go.Scatter(
        x=x, y=y, mode="lines", line=dict(color="#cbd5e1", width=5),
        showlegend=False, hoverinfo="skip"
    ))
    for column, name, color, symbol, position in [
        ("Planned", "Scheduled", "#d97706", "circle", "bottom center"),
        ("Actual", "Actual average", "#2563eb", "diamond", "top center")
    ]:
        fig.add_trace(go.Scatter(
            x=summary[column], y=summary.index, name=name,
            mode="markers+text", marker=dict(size=13, color=color, symbol=symbol),
            text=[f"{v:.2f} days" for v in summary[column]],
            textposition=position, customdata=details,
            hovertemplate=(
                "%{y}<br>" + name + ": %{x:.2f} days"
                "<br>Item rows: %{customdata[0]:,.0f}"
                "<br>Actual minus scheduled: %{customdata[1]:+.2f} days"
                "<br>Late-label rate: %{customdata[2]:.1f}%<extra></extra>"
            )
        ))
    fig.update_layout(
        height=440, template="plotly_white", xaxis_title="Days",
        yaxis=dict(categoryorder="array", categoryarray=summary.index.tolist(), autorange="reversed"),
        xaxis=dict(range=[-0.5, float(summary[["Planned", "Actual"]].max().max()) + 0.8], dtick=1),
        legend=dict(orientation="h", y=1.16), margin=dict(t=80, b=65, l=125, r=40)
    )
    st.plotly_chart(fig, width="stretch", key="berkay_shipping")
    st.caption(
        "Orange = scheduled duration; blue = average recorded duration. "
        "Blue to the right of orange means a longer average duration than scheduled. "
        "Matching averages do not mean every shipment was on time; hover for late-label rates."
    )
    st.caption(
        f"Excluded canceled item rows: {excluded:,}. Durations use the dataset's "
        "shipping-day fields; Same Day values are recorded as 0 or 1 day. "
        "The chart describes this sample, not a guaranteed service duration."
    )


def transaction_summary(df):
    table = pd.crosstab(df["Type"], df["Order Status"])
    order = table.sum(axis=1).sort_values(ascending=False).index
    return table.reindex(order)


def build_transaction_mosaic(df):
    table = transaction_summary(df)
    totals = table.sum(axis=1)
    widths = 100 * totals / totals.sum()
    centers = widths.cumsum() - widths / 2
    percentages = table.div(totals, axis=0) * 100
    styles = {
        "COMPLETE": ("Complete", "#15803d"),
        "CLOSED": ("Closed", "#0f766e"),
        "PROCESSING": ("Processing", "#2563eb"),
        "PENDING": ("Pending", "#7c3aed"),
        "PENDING_PAYMENT": ("Pending payment", "#b45309"),
        "ON_HOLD": ("On hold", "#475569"),
        "PAYMENT_REVIEW": ("Payment review", "#a21caf"),
        "CANCELED": ("Canceled", "#b91c1c"),
        "SUSPECTED_FRAUD": ("Suspected fraud", "#be185d")
    }
    fig = go.Figure()
    for status in table.columns:
        label, color = styles.get(status, (status, "#64748b"))
        pct = percentages[status]
        details = np.column_stack([
            table.index, table[status], totals, widths,
            100 * table[status] / totals.sum()
        ])
        fig.add_trace(go.Bar(
            x=centers, y=pct, width=widths * 0.98,
            name=label, marker=dict(color=color, line=dict(color="white", width=1)),
            text=[f"{label}<br>{v:.1f}%" if v >= 7 else "" for v in pct],
            textposition="inside", insidetextanchor="middle",
            textfont=dict(color="white", size=12), customdata=details,
            hovertemplate=(
                "%{customdata[0]} — " + label +
                "<br>Item rows with this status: %{customdata[1]:,.0f}"
                "<br>Share within this transaction type: %{y:.2f}%"
                "<br>Total item rows of this type: %{customdata[2]:,.0f}"
                "<br>Type's share of all item rows: %{customdata[3]:.2f}%"
                "<br>This combination's share of all item rows: %{customdata[4]:.2f}%"
                "<extra></extra>"
            )
        ))
    fig.update_layout(
        barmode="stack", height=580, template="plotly_white",
        xaxis=dict(
            range=[0, 100], tickmode="array", tickvals=centers.tolist(),
            ticktext=[f"{name}<br>{widths.loc[name]:.1f}% of item rows" for name in table.index],
            title="Transaction type — column width shows its share", showgrid=False
        ),
        yaxis=dict(range=[0, 100], ticksuffix="%", title="Order-status distribution within each type"),
        legend=dict(orientation="h", y=-0.25, x=0),
        margin=dict(t=25, b=145, l=70, r=25)
    )
    return fig, table


def render_transactions(df):
    st.subheader("3. Transaction types and order-status distribution")
    st.caption(
        "Wider columns represent more order-item rows. Colored sections show "
        "the status distribution within each transaction type; each column totals 100%."
    )
    fig, table = build_transaction_mosaic(df)
    st.plotly_chart(fig, width="stretch", key="berkay_transactions")
    st.caption(
        "Hover for counts and percentages. Small sections are retained even when "
        "their labels do not fit. The horizontal widths and vertical percentages "
        "use different denominators: all item rows versus rows within one type."
    )
    st.caption(
        "CASH, DEBIT, PAYMENT and TRANSFER are the dataset's transaction-type labels. "
        "PAYMENT is not relabeled as credit card because the supplied dictionary "
        "does not specify that meaning. CLOSED and COMPLETE remain distinct order statuses. "
        "This snapshot does not measure processing speed or prove that a transaction type causes a status."
    )
    with st.expander("Exact item-row counts"):
        st.dataframe(table, width="stretch")


def render_charts(df):
    st.header("Sales patterns and shipping performance")
    if df.empty:
        st.info("No records match the filters. Please widen your selection.")
        return
    st.caption(
        f"Analyzing {len(df):,} filtered order-item rows. Sidebar filters apply to all charts; "
        "the year selector applies only to Chart 1."
    )
    render_portfolio(df)
    st.divider()
    render_shipping(df)
    st.divider()
    render_transactions(df)
