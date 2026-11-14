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
