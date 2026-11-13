import streamlit as st

def render_charts(df):
    st.header("Can's Visualizations")
    st.info("Add your 3 Plotly visualizations here using the filtered 'df'.")
    # Example:
    # fig = px.bar(df, x="Market", y="Sales")
    # st.plotly_chart(fig, use_container_width=True)
