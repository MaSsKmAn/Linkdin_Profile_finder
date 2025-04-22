import streamlit as st
import json
import re
import csv
from io import StringIO

def process_data(data_list):
    processed = []
    for entry in data_list:
        name_cleaned = re.sub(r"\(.*?\)", "", entry.get("name", "")).strip()
        formatted_name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name_cleaned)

        if entry.get("timezone") and "/" in entry["timezone"]:
            country, state = entry["timezone"].split("/")
        else:
            country, state = "", ""

        company_size = entry.get("company_size") or ""
        company_industry = entry.get("company_industry") or ""
        company_info = re.sub(r'\s+', ' ', f"{company_size} {company_industry}").strip()

        intro = re.sub(r'\s+', ' ', entry.get("intro") or "").strip()
        image = entry.get("image") or ""

        processed.append({
            "name": formatted_name,
            "intro": intro,
            "state": state,
            "country": country,
            "company_info": company_info,
            "image": image
        })

    return processed

st.set_page_config(page_title="User Persona Dashboard", layout="wide")
st.title("🧠 Upload JSON to View User Personas")

uploaded_file = st.file_uploader("Upload your `dataset1.json` file", type=["json"])

if uploaded_file:
    data_list = json.load(uploaded_file)
    processed_data = process_data(data_list)

    st.success(f"Loaded {len(processed_data)} user profiles!")

    for person in processed_data:
        with st.container():
            cols = st.columns([1, 3])
            with cols[0]:
                if person["image"]:
                    st.image(person["image"], width=100)
            with cols[1]:
                st.subheader(person["name"])
                st.caption(f"{person['state']}, {person['country']}")
                st.markdown(f"**Company Info:** {person['company_info']}")
                st.markdown(f"**Intro:** {person['intro']}")
            st.markdown("---")

    search_name = st.text_input("🔍 Enter a name to download the matching profile")

    if search_name:
        matched = [p for p in processed_data if search_name.lower() in p["name"].lower()]

        if matched:
            match_data = matched[0]
            csv_buffer = StringIO()
            writer = csv.DictWriter(csv_buffer, fieldnames=match_data.keys())
            writer.writeheader()
            writer.writerow(match_data)
            st.download_button(
                label="📥 Download Matching Profile as CSV",
                data=csv_buffer.getvalue(),
                file_name=f"{match_data['name'].replace(' ', '_')}_profile.csv",
                mime="text/csv"
            )
        else:
            st.warning("No matching profile found.")
else:
    st.markdown("""
        <div style="background-color: #f0f4ff; padding: 1rem; border-radius: 8px; border-left: 6px solid #1f77b4;">
            <span style="color: #1f77b4; font-weight: 500;">ℹ️ Please upload a JSON file to begin.</span>
        </div>
    """, unsafe_allow_html=True)
