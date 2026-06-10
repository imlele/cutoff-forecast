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
            "Add `openai>=1.40.0` to requirements.txt, commit, and redeploy."
        ) from OPENAI_IMPORT_ERROR

    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def image_to_data_url(image: Image.Image, image_format: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{image_format.lower()};base64,{encoded}"


def split_long_image(
    image: Image.Image,
    max_height: int = 1800,
    overlap: int = 120,
) -> list[Image.Image]:
    width, height = image.size

    if height <= max_height:
        return [image]

    chunks = []
    top = 0

    while top < height:
        bottom = min(top + max_height, height)
        chunk = image.crop((0, top, width, bottom))
        chunks.append(chunk)

        if bottom >= height:
            break

        top = bottom - overlap

    return chunks


def detect_daily_sku_rows_from_chunk(
    image: Image.Image,
    chunk_index: int,
) -> list[dict]:
    client = get_openai_client()
    model = st.secrets.get("OPENAI_OCR_MODEL", "gpt-4.1-mini")

    data_url = image_to_data_url(image)

    prompt = """
You are extracting product sales rows from a long screenshot.

The screenshot may contain Chinese or English columns, such as:
- 商品规格编码 / SKU / sku_no
- 商品名称 / Product Name / sku_name
- 销量 / Quantity / qty
- 销售额 / Amount / Sales

Return JSON only.

Rules:
- Extract visible product rows only.
- Do not invent missing values.
- If sku_no is unclear, use null.
- If sku_name is unclear, use null.
- qty must be a number or null.
- amount must be a number or null.
- Remove currency symbols and commas from amount.
- raw_text should contain the original detected row text.
- If the image chunk has no product sales rows, return {"rows": []}.
"""

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt,
                    },
                    {
                        "type": "input_image",
                        "image_url": data_url,
                    },
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "daily_sku_sales_detection",
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
                                    "sku_no": {
                                        "type": ["string", "null"]
                                    },
                                    "sku_name": {
                                        "type": ["string", "null"]
                                    },
                                    "qty": {
                                        "type": ["number", "null"]
                                    },
                                    "amount": {
                                        "type": ["number", "null"]
                                    },
                                    "raw_text": {
                                        "type": "string"
                                    },
                                },
                                "required": [
                                    "sku_no",
                                    "sku_name",
                                    "qty",
                                    "amount",
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
    image = Image.open(uploaded_file).convert("RGB")
    chunks = split_long_image(image)

    all_rows = []

    progress = st.progress(0)
    status = st.empty()

    for i, chunk in enumerate(chunks):
        status.write(f"Detecting screenshot chunk {i + 1} of {len(chunks)}...")

        rows = detect_daily_sku_rows_from_chunk(
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

    if "raw_text" in df.columns:
        df = df.drop_duplicates(subset=["raw_text"], keep="first")

    return df.reset_index(drop=True)
