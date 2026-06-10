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
    max_height: int = 1600,
    overlap: int = 180,
) -> list[Image.Image]:
    """
    Split long mobile screenshot into overlapping chunks.
    This screenshot is very tall, so chunking improves detection.
    """

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
You are extracting daily product sales data from a long mobile screenshot.

The screenshot title may be:
销售数据

The visible table columns are usually:
- 排名
- 商品名称
- 销量
- 销量占比

Extract all visible product rows.

Return JSON only with this structure:
{
  "rows": [
    {
      "rank": 1,
      "sku_no": null,
      "sku_name": "三倍厚抹",
      "qty": 33,
      "sales_share": 8.4,
      "amount": null,
      "raw_text": "1 三倍厚抹 33 8.4%"
    }
  ]
}

Rules:
- Extract product rows only.
- Do not extract the header row.
- Do not invent SKU codes. If SKU code is not visible, use null.
- 商品名称 goes into sku_name.
- 销量 goes into qty.
- 销量占比 goes into sales_share.
- sales_share should be numeric only. Example: 8.4% becomes 8.4.
- If a value is unclear, use null.
- raw_text should contain the original detected row text.
- If there are duplicate visible rows caused by screenshot overlap, still extract them; the app will remove duplicates later.
- If there are no valid product rows in this chunk, return {"rows": []}.
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
                                    "rank": {
                                        "type": ["integer", "null"]
                                    },
                                    "sku_no": {
                                        "type": ["string", "null"]
                                    },
                                    "sku_name": {
                                        "type": ["string", "null"]
                                    },
                                    "qty": {
                                        "type": ["number", "null"]
                                    },
                                    "sales_share": {
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
                                    "rank",
                                    "sku_no",
                                    "sku_name",
                                    "qty",
                                    "sales_share",
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
    uploaded_file.seek(0)

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

    # Remove overlap duplicates.
    # Prefer rank + sku_name because raw_text may vary slightly.
    if "rank" in df.columns and "sku_name" in df.columns:
        df = df.drop_duplicates(subset=["rank", "sku_name"], keep="first")
    elif "raw_text" in df.columns:
        df = df.drop_duplicates(subset=["raw_text"], keep="first")

    # Sort by ranking if available
    if "rank" in df.columns:
        df = df.sort_values("rank", na_position="last")

    return df.reset_index(drop=True)
