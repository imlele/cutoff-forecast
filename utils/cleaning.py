import pandas as pd


# -----------------------------
# Basic cleaning helpers
# -----------------------------
def normalize_column_name(col: str) -> str:
    col = str(col).strip()
    col = col.replace("\n", " ")
    col = col.replace("\r", " ")
    col = " ".join(col.split())
    return col


def clean_text(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value == "":
        return None

    if value.lower() in ["nan", "none", "null"]:
        return None

    return value


def clean_number(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value == "":
        return None

    if value.lower() in ["nan", "none", "null"]:
        return None

    value = (
        value
        .replace("$", "")
        .replace(",", "")
        .replace("CAD", "")
        .replace("%", "")
        .strip()
    )

    try:
        return float(value)
    except ValueError:
        return None


def parse_hour(value):
    """
    Accepts:
    - 17
    - "17"
    - "17:00"
    - "5 PM"
    - "5PM"
    """
    if pd.isna(value):
        return None

    value = str(value).strip().upper()

    if value == "":
        return None

    if ":" in value:
        value = value.split(":")[0]

    if "PM" in value:
        try:
            num = int(value.replace("PM", "").strip())
            if num != 12:
                return num + 12
            return 12
        except ValueError:
            return None

    if "AM" in value:
        try:
            num = int(value.replace("AM", "").strip())
            if num == 12:
                return 0
            return num
        except ValueError:
            return None

    try:
        return int(float(value))
    except ValueError:
        return None


# -----------------------------
# Tab 1: hourly sales cleaning
# Upload target: sales_hourly
# -----------------------------
def clean_hourly_sales_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final output columns for sales_hourly:

    sale_date
    sale_hour
    sku_no
    sku_name
    qty
    amount
    """

    df = df.copy()
    df.columns = [normalize_column_name(c) for c in df.columns]

    rename_map = {
        # Date
        "date": "sale_date",
        "Date": "sale_date",
        "sale date": "sale_date",
        "Sale Date": "sale_date",
        "销售日期": "sale_date",
        "日期": "sale_date",

        # Hour
        "hour": "sale_hour",
        "Hour": "sale_hour",
        "sale hour": "sale_hour",
        "Sale Hour": "sale_hour",
        "时间": "sale_hour",
        "小时": "sale_hour",
        "营业小时": "sale_hour",

        # SKU no
        "sku_no": "sku_no",
        "sku no": "sku_no",
        "SKU": "sku_no",
        "sku": "sku_no",
        "sku_no商品规格编码": "sku_no",
        "商品规格编码": "sku_no",
        "商品编码": "sku_no",

        # SKU / product name
        "sku_name": "sku_name",
        "sku name": "sku_name",
        "SKU Name": "sku_name",
        "商品名称": "sku_name",
        "sku_name商品名称": "sku_name",
        "产品名称": "sku_name",

        # Quantity
        "qty": "qty",
        "quantity": "qty",
        "Quantity": "qty",
        "qty商品数量": "qty",
        "商品数量": "qty",
        "销量": "qty",
        "销售数量": "qty",

        # Amount
        "amount": "amount",
        "Amount": "amount",
        "sales": "amount",
        "Sales": "amount",
        "销售额": "amount",
        "实收金额": "amount",
        "金额": "amount",
    }

    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

    required_cols = [
        "sale_date",
        "sale_hour",
        "sku_no",
        "sku_name",
        "qty",
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "amount" not in df.columns:
        df["amount"] = None

    df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce").dt.date
    df["sale_hour"] = df["sale_hour"].apply(parse_hour)
    df["sku_no"] = df["sku_no"].apply(clean_text)
    df["sku_name"] = df["sku_name"].apply(clean_text)
    df["qty"] = df["qty"].apply(clean_number)
    df["amount"] = df["amount"].apply(clean_number)

    df = df[
        [
            "sale_date",
            "sale_hour",
            "sku_no",
            "sku_name",
            "qty",
            "amount",
        ]
    ]

    df = df.dropna(
        subset=[
            "sale_date",
            "sale_hour",
            "sku_no",
            "qty",
        ]
    )

    df = df.drop_duplicates(
        subset=[
            "sale_date",
            "sale_hour",
            "sku_no",
        ],
        keep="last",
    )

    return df.reset_index(drop=True)


# -----------------------------
# Tab 2: daily SKU screenshot cleaning
# Upload target: daily_sku_sales
# Screenshot columns:
# 排名 | 商品名称 | 销量 | 销量占比
# -----------------------------
def clean_daily_sku_sales_df(
    df: pd.DataFrame,
    sale_date,
    source_file: str,
) -> pd.DataFrame:
    """
    Final output columns for daily_sku_sales:

    sale_date
    rank
    product_name
    qty
    sales_share
    raw_text
    source_file
    """

    df = df.copy()

    expected_columns = [
        "rank",
        "product_name",
        "qty",
        "sales_share",
        "raw_text",
    ]

    for col in expected_columns:
        if col not in df.columns:
            df[col] = None

    df["sale_date"] = sale_date
    df["source_file"] = source_file

    df["rank"] = df["rank"].apply(clean_number)
    df["product_name"] = df["product_name"].apply(clean_text)
    df["qty"] = df["qty"].apply(clean_number)
    df["sales_share"] = df["sales_share"].apply(clean_number)
    df["raw_text"] = df["raw_text"].apply(clean_text)

    # Convert rank to integer if possible
    df["rank"] = df["rank"].apply(
        lambda x: int(x) if x is not None else None
    )

    df = df[
        [
            "sale_date",
            "rank",
            "product_name",
            "qty",
            "sales_share",
            "raw_text",
            "source_file",
        ]
    ]

    # Product name and qty are required for this screenshot format
    df = df.dropna(
        subset=[
            "sale_date",
            "product_name",
            "qty",
        ]
    )

    # Remove duplicates caused by long screenshot overlap
    df = df.drop_duplicates(
        subset=[
            "sale_date",
            "product_name",
        ],
        keep="last",
    )

    # Sort by ranking
    if "rank" in df.columns:
        df = df.sort_values(
            by="rank",
            na_position="last",
        )

    return df.reset_index(drop=True)


# -----------------------------
# Convert DataFrame to Supabase rows
# -----------------------------
def dataframe_to_supabase_rows(df: pd.DataFrame) -> list[dict]:
    """
    Converts pandas DataFrame into list[dict] for Supabase insert/upsert.

    Handles:
    - NaN -> None
    - date/datetime -> ISO string
    - pandas/numpy numbers -> Python numbers
    """

    rows = []

    for _, row in df.iterrows():
        item = {}

        for col in df.columns:
            value = row[col]

            if pd.isna(value):
                item[col] = None
            elif hasattr(value, "isoformat"):
                item[col] = value.isoformat()
            elif isinstance(value, float):
                item[col] = float(value)
            elif isinstance(value, int):
                item[col] = int(value)
            else:
                item[col] = value

        rows.append(item)

    return rows
