import streamlit as st
from supabase import create_client, Client


@st.cache_resource
def get_supabase_client() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


def insert_rows(table_name: str, rows: list[dict]):
    """
    Bulk insert rows into Supabase.
    """
    if not rows:
        return {"inserted": 0, "data": []}

    supabase = get_supabase_client()

    response = (
        supabase
        .table(table_name)
        .insert(rows)
        .execute()
    )

    return response


def upsert_rows(table_name: str, rows: list[dict], on_conflict: str):
    """
    Upsert rows into Supabase.
    Use this if you created a unique index.
    Example on_conflict:
      - sales_hourly: "sale_date,sale_hour,sku_no"
      - daily_sku_sales: "sale_date,sku_no"
    """
    if not rows:
        return {"inserted": 0, "data": []}

    supabase = get_supabase_client()

    response = (
        supabase
        .table(table_name)
        .upsert(rows, on_conflict=on_conflict)
        .execute()
    )

    return response
