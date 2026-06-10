import streamlit as st
import pandas as pd

from utils.cleaning import (
    clean_hourly_sales_df,
    dataframe_to_supabase_rows,
)
from utils.supabase_client import insert_rows, upsert_rows


def read_uploaded_table(uploaded_file) -> pd.DataFrame:
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        return pd.read_excel(uploaded_file)

    raise ValueError("Unsupported file type. Please upload CSV or Excel.")


def render_tab1_hourly_upload():
    st.header("Tab 1: Upload Hourly Sales")
    st.caption("This uploads cleaned hourly data into `sales_hourly`.")

    uploaded_file = st.file_uploader(
        "Upload hourly sales CSV or Excel",
        type=["csv", "xlsx", "xls"],
        key="hourly_upload_file",
    )

    upload_mode = st.radio(
        "Upload mode",
        options=[
            "Insert new rows",
            "Upsert / replace duplicate rows",
        ],
        index=1,
        key="hourly_upload_mode",
    )

    if uploaded_file is None:
        st.info("Upload a CSV or Excel file to start.")
        return

    try:
        raw_df = read_uploaded_table(uploaded_file)

        st.subheader("Raw uploaded data")
        st.dataframe(raw_df, use_container_width=True)

        cleaned_df = clean_hourly_sales_df(raw_df)
        cleaned_df["source_file"] = uploaded_file.name

        st.subheader("Cleaned hourly data")
        st.dataframe(cleaned_df, use_container_width=True)

        st.success(f"Ready to upload {len(cleaned_df)} rows to `sales_hourly`.")

        edited_df = st.data_editor(
            cleaned_df,
            use_container_width=True,
            num_rows="dynamic",
            key="hourly_cleaned_editor",
        )

        if st.button("Confirm and upload to sales_hourly", key="upload_hourly_confirm"):
            rows = dataframe_to_supabase_rows(edited_df)

            table_name = st.secrets.get("SALES_HOURLY_TABLE", "sales_hourly")

            if upload_mode == "Upsert / replace duplicate rows":
                response = upsert_rows(
                    table_name=table_name,
                    rows=rows,
                    on_conflict="sale_date,sale_hour,sku_no",
                )
            else:
                response = insert_rows(
                    table_name=table_name,
                    rows=rows,
                )

            st.success(f"Uploaded {len(rows)} rows to `{table_name}`.")
            st.write(response)

    except Exception as e:
        st.error("Failed to process hourly upload.")
        st.exception(e)
