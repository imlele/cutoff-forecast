import pandas as pd


def normalize_column_name(col: str) -> str:
    col = str(col).strip()
    col = col.replace("\n", " ")
    col = col.replace("\r", " ")
    col = " ".join(col.split())
    return col


def clean_number(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value == "":
        return None

    value = (
        value
        .replace("$", "")
        .replace(",", "")
        .replace("CAD", "")
        .strip()
    )

    try:
        return float(value)
    except ValueError:
        return None


def clean_text(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value == "":
        return None

    return value


def parse_hour(value):
    """
    Accepts:
    - 17
    - "17"
    - "17:00"
    - "5 PM"
    """
    if pd.isna(value):
        return None

    value = str(value).strip().upper()

    if value == "":
        return None

    if ":" in value:
        value = value.split(":")[0]

    if "PM" in value:
        num = int(value.replace("PM", "").strip())
        if num != 12:
            return num + 12
        return 12

    if "AM" in value:
        num = int(value.replace("AM", "").strip())
        if num == 12:
            return 0
        return num

    try:
        return int(float(value))
    except ValueError:
        return None


def clean_hourly_sales_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expected final output columns:
    sale_date, sale_hour, sku_no, sku_name, qty, amount
    """

    df = df.copy()
    df.columns = [normalize_column_name(c) for c in df.columns]

    rename_map = {
        "date": "sale_date",
        "Date": "sale_date",
        "sale date": "sale_date",
        "Sale Date": "sale_date",
        "销售日期": "sale_date",

        "hour": "sale_hour",
        "Hour": "sale_hour",
        "sale hour": "sale_hour",
        "时间": "sale_hour",
        "小时": "sale_hour",

        "sku_no": "sku_no",
        "sku no": "sku_no",
        "SKU": "sku_no",
        "sku": "sku_no",
        "sku_no商品规格编码": "sku_no",
        "商品规格编码": "sku_no",

        "sku_name": "sku_name",
        "sku name": "sku_name",
        "商品名称": "sku_name",
        "sku_name商品名称": "sku_name",

        "qty": "qty",
        "quantity": "qty",
        "qty商品数量": "qty",
        "商品数量": "qty",
        "销量": "qty",

        "amount": "amount",
        "sales": "amount",
        "销售额": "amount",
        "实收金额": "amount",
    }

    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

    required_cols = ["sale_date", "sale_hour", "sku_no", "sku_name", "qty"]
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

    df = df[["sale_date", "sale_hour", "sku_no", "sku_name", "qty", "amount"]]

    df = df.dropna(subset=["sale_date", "sale_hour", "sku_no", "qty"])
    df = df.drop_duplicates(subset=["sale_date", "sale_hour", "sku_no"], keep="last")

    return df.reset_index(drop=True)


def clean_daily_sku_sales_df(df: pd.DataFrame, sale_date, source_file: str) -> pd.DataFrame:
    """
    Expected final output columns:
    sale_date, sku_no, sku_name, qty, amount, raw_text, source_file
    """

    df = df.copy()

    for col in ["sku_no", "sku_name", "raw_text"]:
        if col not in df.columns:
            df[col] = None

    if "qty" not in df.columns:
        df["qty"] = None

    if "amount" not in df.columns:
        df["amount"] = None

    df["sale_date"] = sale_date
    df["source_file"] = source_file

    df["sku_no"] = df["sku_no"].apply(clean_text)
    df["sku_name"] = df["sku_name"].apply(clean_text)
    df["qty"] = df["qty"].apply(clean_number)
    df["amount"] = df["amount"].apply(clean_number)
    df["raw_text"] = df["raw_text"].apply(clean_text)

    df = df[[
        "sale_date",
        "sku_no",
        "sku_name",
        "qty",
        "amount",
        "raw_text",
        "source_file",
    ]]

    df = df.dropna(subset=["sale_date", "sku_no", "qty"])
    df = df.drop_duplicates(subset=["sale_date", "sku_no"], keep="last")

    return df.reset_index(drop=True)


def dataframe_to_supabase_rows(df: pd.DataFrame) -> list[dict]:
    rows = []

    for _, row in df.iterrows():
        item = {}

        for col in df.columns:
            value = row[col]

            if pd.isna(value):
                item[col] = None
            elif hasattr(value, "isoformat"):
                item[col] = value.isoformat()
            else:
                item[col] = value

        rows.append(item)

    return rows
