import streamlit as st

from tabs.tab1_hourly_upload import render_tab1_hourly_upload
from tabs.tab2_screenshot_upload import render_tab2_screenshot_upload
from tabs.tab3_predict import render_tab3_predict


st.set_page_config(
    page_title="Sales Upload & Forecast App",
    page_icon="📊",
    layout="wide",
)


def main():
    st.title("Sales Upload & Forecast App")

    tab1, tab2, tab3 = st.tabs([
        "1. Upload Hourly Data",
        "2. Upload Screenshot",
        "3. Predict",
    ])

    with tab1:
        render_tab1_hourly_upload()

    with tab2:
        render_tab2_screenshot_upload()

    with tab3:
        render_tab3_predict()


if __name__ == "__main__":
    main()
