import base64
import io
import json
from PIL import Image
import streamlit as st
import pandas as pd

try:
    from openai import OpenAI
except ImportError as e:
    OpenAI = None
    OPENAI_IMPORT_ERROR = e
else:
    OPENAI_IMPORT_ERROR = None


@st.cache_resource
def get_openai_client():
    if OpenAI is None:
        raise ImportError(
            "OpenAI package is not installed correctly. "
            "Add openai>=1.40.0 to requirements.txt and redeploy."
        ) from OPENAI_IMPORT_ERROR

    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def split_long_image(image: Image.Image, max_height: int = 1500, overlap: int = 200):
    width, height = image.size

    if height <= max_height:
        return [image]

    chunks = []
    top = 0

    while top < height:
        bottom = min(top + max_height, height)
        chunks.append(image.crop((0, top, width, bottom)))

        if bottom >= height:
            break

        top = bottom - overlap

    return chunks


def detect_sales_rows_from_chunk(image: Image.Image, chunk_index: int) -> list[dict]:
    client = get_openai_client()
    model = st.secrets.get("OPENAI_OCR_MODEL", "gpt-4.1-mini")

    data_url = image_to_data_url(image)

    prompt = """
You are extracting sales data from a Chinese mobile screenshot.

The table columns are:
排名, 商品名称, 销量, 销量占比

Return JSON only in this exact format:

{
  "rows": [
    {
      "rank": 1,
      "product_name": "三倍厚抹",
      "qty": 33,
      "sales_share": 8.4,
      "raw_text": "1 三倍厚抹 33 8.4%"
    }
  ]
}

Rules:
- Extract product rows only.
- Do not extract the header.
- rank = 排名.
- product_name = 商品名称.
- qty = 销量.
- sales_share = 销量占比.
- sales_share must be numeric only. Example: 8.4% becomes 8.4.
- If a value is unclear, use null.
- Do not invent data.
- If no rows exist, return {"rows": []}.
"""

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "daily_sku_sales_rows",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "rows": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "rank": {"type": ["integer", "null"]},
                                    "product_name": {"type": ["string", "null"]},
                                    "qty": {"type": ["number", "null"]},
                                    "sales_share": {"type": ["number", "null"]},
                                    "raw_text": {"type": "string"},
                                },
                                "required": [
                                    "rank",
                                    "product_name",
                                    "qty",
                                    "sales_share",
                                    "raw_text",
                                ],
                            },
                        }
                    },
                    "required": ["rows"],
                },
                "strict": True,
            }
        },
    )

    result = json.loads(response.output_text)
    rows = result.get("rows", [])

    for row in rows:
        row["chunk_index"] = chunk_index

    return rows


def detect_daily_sku_rows_from_long_screenshot(uploaded_file) -> pd.DataFrame:
    uploaded_file.seek(0)
    image = Image.open(uploaded_file).convert("RGB")

    chunks = split_long_image(image)

    all_rows = []

    progress = st.progress(0)
    status = st.empty()

    for i, chunk in enumerate(chunks):
        status.write(f"Detecting image part {i + 1} of {len(chunks)}...")

        rows = detect_sales_rows_from_chunk(
            image=chunk,
            chunk_index=i + 1,
        )

        all_rows.extend(rows)
        progress.progress((i + 1) / len(chunks))

    status.empty()
    progress.empty()

    df = pd.DataFrame(all_rows)

    if df.empty:
        return df

    df = df.drop_duplicates(
        subset=["rank", "product_name"],
        keep="first",
    )

    df = df.sort_values("rank", na_position="last")

    return df.reset_index(drop=True)
