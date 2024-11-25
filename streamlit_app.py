import streamlit as st
from apps import upload, upload2  # import both upload and upload2 apps

st.set_page_config(page_title="Streamlit Geospatial", layout="wide")

# Two apps for uploading
apps = [
    {"func": upload2.app, "title": "Fault Detection Model", "icon": "cloud-upload"},
    {"func": upload.app, "title": "Folium Map", "icon": "cloud-upload"},
]

# Titles and icons for both apps
titles = [app["title"] for app in apps]
icons = [app["icon"] for app in apps]

# Sidebar
with st.sidebar:
    selected = st.selectbox(
        "Main Menu",
        options=titles,
        index=0,  # Default selected is the first app
    )

    # st.sidebar.title("Test")
    # st.sidebar.info(
    #     """
    # Sidebar content.
    # """
    # )

# Run the selected app
for app in apps:
    if app["title"] == selected:
        app["func"]()
        break
