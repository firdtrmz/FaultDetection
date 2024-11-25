import streamlit as st
import PIL
import io
import zipfile

# Import settings and helper modules from your YOLOv8 project
import settings
import helper


def extract_zip_in_memory(zip_file):
    """
    Extract files from a ZIP file in memory.
    """
    extracted_files = []
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            with zip_ref.open(file_info) as file:
                extracted_files.append((file_info.filename, file.read()))
    return extracted_files


def app():
    # Main page heading
    st.title("Fault Detection on Power Distribution Asset using YOLOv8")

    # Sidebar configuration
    st.sidebar.header("ML Model Config")

    # Model options (Detection only)
    model_type = st.sidebar.selectbox(
        "Select Task", ['Detection'])

    confidence = float(st.sidebar.slider(
        "Select Model Confidence", 25, 100, 40)) / 100

    # Load pre-trained Detection model
    try:
        detection_model = helper.load_model(settings.DETECTION_MODEL)
    except Exception as ex:
        st.error(f"Unable to load detection model. Check the specified path: {settings.DETECTION_MODEL}")
        st.error(ex)
        return

    st.sidebar.header("Image Config")

    # Only keep the "Image" source in the sidebar
    source_radio = st.sidebar.radio(
        "Select Source", [settings.IMAGE])  # Only show Image source option

    if source_radio == settings.IMAGE:
        zip_file = st.sidebar.file_uploader(
            "Upload a ZIP file containing images...", type="zip")

        if zip_file:
            try:
                # Extract images from the ZIP file in memory
                extracted_files = extract_zip_in_memory(zip_file)

                # Process and detect images
                detected_images = []
                for file_name, file_data in extracted_files:
                    if file_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                        try:
                            uploaded_image = PIL.Image.open(io.BytesIO(file_data))
                            # Perform detection
                            res = detection_model.predict(uploaded_image, conf=confidence)
                            detected_image = res[0].plot()[:, :, ::-1]
                            detected_images.append((file_name, detected_image))
                        except Exception as ex:
                            st.error(f"Error processing image {file_name}: {str(ex)}")

                # Display all detected images in a 3x3 collage
                if detected_images:
                    st.write("Detected Images:")

                    # Define grid size (3x3)
                    grid_size = 3

                    # Create a grid layout
                    num_images = len(detected_images)
                    rows = (num_images + grid_size - 1) // grid_size  # Calculate number of rows needed

                    for i in range(rows):
                        cols = st.columns(grid_size)
                        for j in range(grid_size):
                            index = i * grid_size + j
                            if index < num_images:
                                file_name, detected_image = detected_images[index]
                                with cols[j]:
                                    st.image(detected_image, caption=f'Detected Image: {file_name}',
                                             use_column_width=True)
                            else:
                                with cols[j]:
                                    st.write("")  # Empty cell

            except Exception as ex:
                st.error(f"An error occurred while extracting ZIP file: {str(ex)}")

