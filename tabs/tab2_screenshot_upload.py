import streamlit as st
from datetime import date
from PIL import Image

from utils.openai_vision import detect_daily_sku_rows_from_long_screenshot
from utils.cleaning import (
    clean_daily_sku_sales_df,
    dataframe_to_supabase_rows,
)
from utils.supabase_client import insert_rows, upsert_rows


def render_tab2_screenshot_upload():
    st.header("Tab 2: Upload Daily SKU Sales Screenshot")

    st.caption(
        "Upload the sales ranking screenshot image. "
        "OpenAI will detect: 排名, 商品名称, 销量, 销量占比. "
        "You can review and edit the detected data before uploading to `daily_sku_sales`."
    )

    # -----------------------------
    # Date input
    # -----------------------------
    sale_date = st.date_input(
        "Sales date",
        value=date.today(),
        key="daily_sku_sale_date",
    )

    # -----------------------------
    # Image uploader only
    # -----------------------------
    uploaded_image = st.file_uploader(
        "Upload screenshot image",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=False,
        key="daily_sku_screenshot_image",
    )

    upload_mode = st.radio(
        "Upload mode",
        options=[
            "Upsert / replace duplicate rows",
            "Insert new rows only",
        ],
        index=0,
        key="daily_sku_upload_mode",
    )

    st.info(
        "Expected screenshot columns: 排名 | 商品名称 | 销量 | 销量占比"
    )

    if uploaded_image is None:
        st.warning("Please upload a screenshot image to start.")
        return

    # -----------------------------
    # Show uploaded image
    # -----------------------------
    try:
        uploaded_image.seek(0)
        image = Image.open(uploaded_image).convert("RGB")

        st.subheader("Uploaded screenshot preview")
        st.image(
            image,
            caption=f"Uploaded image: {uploaded_image.name}",
            use_container_width=True,
        )

        # Important: reset file pointer for OpenAI detection later
        uploaded_image.seek(0)

    except Exception as e:
        st.error("The uploaded file is not a valid image.")
        st.exception(e)
        return

    # -----------------------------
    # Detect with OpenAI
    # -----------------------------
    if st.button("Detect data with OpenAI", key="detect_daily_sku_btn"):
        try:
            uploaded_image.seek(0)

            with st.spinner("Detecting sales rows from screenshot..."):
                detected_df = detect_daily_sku_rows_from_long_screenshot(
                    uploaded_image
                )

            if detected_df.empty:
                st.warning(
                    "No valid rows were detected. "
                    "Try uploading a clearer screenshot or cropping closer to the table."
                )
                return

            cleaned_df = clean_daily_sku_sales_df(
                detected_df,
                sale_date=sale_date,
                source_file=uploaded_image.name,
            )

            if cleaned_df.empty:
                st.warning(
                    "Rows were detected, but none passed cleaning. "
                    "Please check the screenshot format."
                )
                st.dataframe(detected_df, use_container_width=True)
                return

            st.session_state["daily_sku_detected_df"] = cleaned_df
            st.session_state["daily_sku_source_file"] = uploaded_image.name

            st.success(f"Detected {len(cleaned_df)} valid rows.")

        except Exception as e:
            st.error("OpenAI image detection failed.")
            st.exception(e)
            return

    # -----------------------------
    # Stop here until detection done
    # -----------------------------
    if "daily_sku_detected_df" not in st.session_state:
        return

    st.divider()

    # -----------------------------
    # Review detected data
    # -----------------------------
    st.subheader("Review and edit detected data")

    st.caption(
        "Final upload columns: sale_date, rank, product_name, qty, sales_share, raw_text, source_file"
    )

    edited_df = st.data_editor(
        st.session_state["daily_sku_detected_df"],
        use_container_width=True,
        num_rows="dynamic",
        key="daily_sku_detected_editor",
        column_config={
            "sale_date": st.column_config.DateColumn(
                "sale_date",
                help="Sales date for this screenshot",
            ),
            "rank": st.column_config.NumberColumn(
                "rank",
                help="排名",
                step=1,
                format="%d",
            ),
            "product_name": st.column_config.TextColumn(
                "product_name",
                help="商品名称",
            ),
            "qty": st.column_config.NumberColumn(
                "qty",
                help="销量",
                step=1,
            ),
            "sales_share": st.column_config.NumberColumn(
                "sales_share",
                help="销量占比, numeric only. Example: 8.4 means 8.4%",
                format="%.2f",
            ),
            "raw_text": st.column_config.TextColumn(
                "raw_text",
                help="Original detected row text",
            ),
            "source_file": st.column_config.TextColumn(
                "source_file",
                help="Original screenshot file name",
            ),
        },
    )

    # -----------------------------
    # Final preview
    # -----------------------------
    st.subheader("Final upload preview")
    st.dataframe(edited_df, use_container_width=True)

    st.write(f"Rows ready to upload: **{len(edited_df)}**")

    # -----------------------------
    # Upload to Supabase
    # -----------------------------
    if st.button(
        "Confirm and upload to daily_sku_sales",
        key="upload_daily_sku_confirm",
        type="primary",
    ):
        try:
            if edited_df.empty:
                st.warning("There is no data to upload.")
                return

            rows = dataframe_to_supabase_rows(edited_df)

            table_name = st.secrets.get(
                "DAILY_SKU_SALES_TABLE",
                "daily_sku_sales",
            )

            with st.spinner(f"Uploading {len(rows)} rows to `{table_name}`..."):
                if upload_mode == "Upsert / replace duplicate rows":
                    response = upsert_rows(
                        table_name=table_name,
                        rows=rows,
                        on_conflict="sale_date,product_name",
                    )
                else:
                    response = insert_rows(
                        table_name=table_name,
                        rows=rows,
                    )

            st.success(f"Uploaded {len(rows)} rows to `{table_name}` successfully.")
            st.write(response)

        except Exception as e:
            st.error("Failed to upload daily SKU sales to Supabase.")
            st.exception(e)
