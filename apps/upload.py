import os
import geopandas as gpd
import streamlit as st
import fiona  # For reading different file formats like KML
import pandas as pd  # To concatenate GeoDataFrames
import json  # To handle JSON/GeoJSON
import requests  # For downloading images
from PIL import Image
import io

# Import settings and helper modules from your YOLOv8 project
import settings
import helper

def save_uploaded_file(file_content, file_name):
    """ Save the uploaded file to a temporary directory. """
    import tempfile
    import uuid

    _, file_extension = os.path.splitext(file_name)
    file_id = str(uuid.uuid4())
    file_path = os.path.join(tempfile.gettempdir(), f"{file_id}{file_extension}")

    with open(file_path, "wb") as file:
        file.write(file_content.getbuffer())

    return file_path

def extract_image_link(description):
    """ Extracts the image link from the 'description' field if available. """
    import re
    image_match = re.search(r'<img src="([^"]+)"', description)
    if image_match:
        return image_match.group(1)
    return None

def display_fault_detection(conf_threshold):
    """ Handles fault detection for the specified image path and displays the result in the sidebar. """
    st.sidebar.header("Fault Detection Results")

    image_path = "Lundu-Sematan H7L5 (3)/Lundu-Sematan H7L5/rotten/image (37).png"

    # Load the original image from the specified path
    try:
        original_image = Image.open(image_path)
        st.sidebar.image(original_image, caption="Original Image", use_column_width=True)
    except Exception as e:
        st.sidebar.error(f"Failed to load image: {str(e)}")
        return

    # Load pre-trained Detection model
    try:
        detection_model = helper.load_model(settings.DETECTION_MODEL)
    except Exception as ex:
        st.sidebar.error(f"Unable to load detection model. Check the specified path: {settings.DETECTION_MODEL}")
        st.sidebar.error(ex)
        return

    # Perform fault detection on the image with the specified confidence threshold
    try:
        res = detection_model.predict(image_path, conf=conf_threshold)  # Use confidence threshold from slider
        detected_image = res[0].plot()[:, :, ::-1]  # Process detection results
        st.sidebar.image(detected_image, caption="Detected Image", use_column_width=True)
        st.sidebar.write(f"Confidence Threshold: {conf_threshold}")
    except Exception as ex:
        st.sidebar.error(f"Error processing image: {str(ex)}")

def app():
    st.title("Upload Vector Data")

    # Coordinates for Sarawak, Malaysia
    sarawak_lat = 1.5533
    sarawak_lon = 110.3592

    # Set a zoom level for Sarawak
    zoom_level = 10  # Zoom level for Sarawak

    # File uploader goes outside of the column block to stay above the map
    data = st.file_uploader(
        "Upload vector datasets", type=["geojson", "json", "kml", "zip", "tab"], accept_multiple_files=True
    )

    row1_col1, row1_col2 = st.columns([2, 1])  # Create column layout for map and options
    width = 1700
    height = 600

    with row1_col2:
        container = st.container()

        if data:
            all_gdfs = []  # List to hold all GeoDataFrames

            # Loop through each uploaded file
            for file in data:
                file_path = save_uploaded_file(file, file.name)
                layer_name = os.path.splitext(file.name)[0]

                with row1_col1:
                    try:
                        # Handling GeoJSON and JSON files
                        if file_path.lower().endswith((".geojson", ".json")):
                            with open(file_path) as f:
                                geojson_data = json.load(f)  # Load the GeoJSON data
                                gdf = gpd.GeoDataFrame.from_features(
                                    geojson_data["features"])  # Convert to GeoDataFrame
                        elif file_path.lower().endswith(".kml"):
                            # Enable KML driver in fiona
                            fiona.drvsupport.supported_drivers["KML"] = "rw"
                            gdf = gpd.read_file(file_path, driver="KML")
                        else:
                            gdf = gpd.read_file(file_path)
                    except Exception as e:
                        st.error(f"Failed to load the file {file.name}: {str(e)}")
                        continue  # Continue with the next file

                    # Add GeoDataFrame to the list
                    all_gdfs.append((gdf, layer_name))

            if all_gdfs:
                # Combine all GeoDataFrames into one
                combined_gdf = gpd.GeoDataFrame(pd.concat([g[0] for g in all_gdfs], ignore_index=True))

                try:
                    lon, lat = combined_gdf.geometry.centroid.x.mean(), combined_gdf.geometry.centroid.y.mean()
                except Exception as e:
                    st.error(f"Failed to calculate centroids: {str(e)}")
                    return

                # Set map centered at Sarawak with zoom level and hybrid basemap
                import leafmap.foliumap as leafmap
                m = leafmap.Map(center=(sarawak_lat, sarawak_lon), zoom=zoom_level, draw_export=True)
                m.add_basemap("HYBRID")  # Set the basemap to hybrid (satellite + labels)

                # Add each GeoDataFrame as a layer on the map
                for gdf, layer_name in all_gdfs:
                    for _, row in gdf.iterrows():
                        description = row.get("description", "")
                        image_link = row.get("gx_media_links", "")
                        name = row.get("Name", "")

                        popup_content = ""
                        if image_link:
                            # Display image only in popup, no detect fault link
                            popup_content += f'<img src="{image_link}" width="300" height="auto"/><br>'

                        popup_content += f'{description}<br><strong>{name}</strong>'

                        color = 'red' if 'Rotten' in description else 'blue'  # Check for description

                        # Add marker to map
                        m.add_marker(
                            location=[row.geometry.y, row.geometry.x],
                            popup=popup_content,
                            layer_name=layer_name,
                            icon_color=color
                        )

                        # Check if this is the marker you want to run the detection for
                        if name == "1.6222674 N , 111.4634386 E":
                            # Add confidence threshold slider
                            confidence_threshold = st.sidebar.slider(
                                "Confidence Threshold", min_value=0.0, max_value=1.0, value=0.4, step=0.05
                            )
                            st.sidebar.button(
                                "Detect Fault",
                                on_click=display_fault_detection,
                                args=(confidence_threshold,)
                            )

                # Zoom to the combined GeoDataFrame
                m.zoom_to_gdf(combined_gdf)
                m.to_streamlit(width=width, height=height)
            else:
                st.error("No valid GeoDataFrames were loaded.")
        else:
            with row1_col1:
                # Map centered at Sarawak with zoom level and hybrid basemap
                import leafmap.foliumap as leafmap
                m = leafmap.Map(center=(sarawak_lat, sarawak_lon), zoom=zoom_level)
                m.add_basemap("HYBRID")  # Set the basemap to hybrid
                m.to_streamlit(width=width, height=height)
