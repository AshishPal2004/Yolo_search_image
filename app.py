import streamlit as st
import sys
import time
from PIL import Image, ImageDraw, ImageFont
import base64
import json
import io
import os
import zipfile
from pathlib import Path

# Add project root to system path
sys.path.append(str(Path(__file__).parent))

from src.inference import YOLOv11Inference
from src.utils import save_metadata, load_metadata, get_unique_classes_counts
from src.config import RAW_DIR, PROCESSED_DIR

def img_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def init_session_state():
    session_defaults = {
        "metadata": None,
        "unique_classes": [],
        "count_options": {},
        "search_results": [],
        "search_params": {
            "search_mode": "Any of selected classes (OR)",
            "selected_classes": [],
            "thresholds": {}
        },
        "show_boxes": True,
        "grid_columns": 3,
        "highlight_matches": True
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

st.set_page_config(page_title="YOLOv11 Search App", layout="wide")
st.title("Computer Vision Powered Search Application")

# CSS Styling
st.markdown("""
<style>
.image-card { border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1); transition: all 0.3s ease; margin-bottom: 20px; background: #f8f9fa; }
.image-card:hover { transform: translateY(-3px); box-shadow: 0 6px 16px rgba(0,0,0,0.15); }
.image-container { position: relative; width: 100%; aspect-ratio: 4/3; }
.image-container img { width: 100%; height: 100%; object-fit: cover; }
.meta-overlay { padding: 10px; background: rgba(0,0,0,0.85); color: white; font-size: 13px; line-height: 1.4; }
</style>
""", unsafe_allow_html=True)

option = st.radio("Choose an option:", ("Process new images", "Load existing metadata"), horizontal=True)

if option == "Process new images":
    with st.expander("Process new images", expanded=True):
        input_method = st.radio("Select Source:", ("Upload Images / ZIP (Saves to data/raw)", "Local Directory Path"), horizontal=True)
        model_path = st.text_input("Model weights path:", "yolo11m.pt") # Using the yolo11m.pt from your root
        
        target_dir = None
        dataset_name = "default_dataset"

        if input_method == "Upload Images / ZIP (Saves to data/raw)":
            dataset_name = st.text_input("Dataset Name (Folder to create):", value="uploaded_batch").strip()
            uploaded_files = st.file_uploader("Upload images or .zip", type=["jpg", "jpeg", "png", "zip"], accept_multiple_files=True)

            if uploaded_files:
                target_dir = RAW_DIR / dataset_name
                target_dir.mkdir(parents=True, exist_ok=True)

                for uploaded_file in uploaded_files:
                    if uploaded_file.name.endswith(".zip"):
                        with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
                            zip_ref.extractall(target_dir)
                    else:
                        file_path = target_dir / uploaded_file.name
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                
                target_dir = str(target_dir)
                st.info(f"Files saved to `data/raw/{dataset_name}`")

        else:
            target_dir = st.text_input("Image directory path:", placeholder="e.g., E:/dataset/images")
            if target_dir:
                dataset_name = Path(target_dir).name

        if st.button("Start Inference"):
            if target_dir and os.path.exists(target_dir):
                try:
                    with st.spinner("Running object detection..."):
                        inferencer = YOLOv11Inference(model_path)
                        metadata = inferencer.process_directory(target_dir)

                        
                        metadata_path = save_metadata(metadata, dataset_name) 
                        
                        # 1. Update metadata state
                        st.session_state.metadata = metadata
                        st.session_state.unique_classes, st.session_state.count_options = get_unique_classes_counts(metadata)
                        
                        # 2. Reset the search engine state completely
                        st.session_state.search_results = []
                        st.session_state.search_params = {
                            "search_mode": "Any of selected classes (OR)",
                            "selected_classes": [],
                            "thresholds": {}
                        }

                        st.success(f"Processed {len(metadata)} images! Refreshing UI...")
                        time.sleep(1) 
                        st.rerun()    # Force the UI to refresh with the new data
                        
                except Exception as e:
                    st.error(f"Error during inference: {str(e)}")
            else:
                st.warning("Please specify a valid path or upload files.")

else:
    with st.expander("Load Existing Metadata", expanded=True):
        uploaded_meta = st.file_uploader("Upload metadata.json file", type=["json"])
        local_meta_path = st.text_input("OR enter Metadata file path:", placeholder="data/processed/dataset_name/metadata.json")

        if st.button("Load Metadata"):
            try:
                with st.spinner("Loading Metadata..."):
                    if uploaded_meta:
                        metadata = json.load(uploaded_meta)
                    elif local_meta_path:
                        metadata = load_metadata(local_meta_path)
                    else:
                        st.warning("Please upload or link a metadata file.")
                        metadata = None

                    if metadata:
                        # 1. Update metadata state
                        st.session_state.metadata = metadata
                        st.session_state.unique_classes, st.session_state.count_options = get_unique_classes_counts(metadata)
                        
                        # 2. Reset the search engine state completely
                        st.session_state.search_results = []
                        st.session_state.search_params = {
                            "search_mode": "Any of selected classes (OR)",
                            "selected_classes": [],
                            "thresholds": {}
                        }
                        
                        st.success(f"Loaded metadata for {len(metadata)} images. Refreshing UI...")
                        time.sleep(1) # Pause briefly so you can read the success message
                        st.rerun()    # Force the UI to refresh with the new data
            except Exception as e:
                st.error(f"Error loading metadata: {str(e)}")

# Search Engine Section
if st.session_state.metadata:
    st.header("🔍 Search Engine")

    with st.container():
        st.session_state.search_params["search_mode"] = st.radio("Search mode:", ("Any of selected classes (OR)", "All selected classes (AND)"), horizontal=True)
        st.session_state.search_params["selected_classes"] = st.multiselect("Classes to search for:", options=st.session_state.unique_classes)

        if st.session_state.search_params["selected_classes"]:
            st.subheader("Count Thresholds (optional)")
            cols = st.columns(len(st.session_state.search_params["selected_classes"]))
            for i, cls in enumerate(st.session_state.search_params["selected_classes"]):
                with cols[i]:
                    st.session_state.search_params["thresholds"][cls] = st.selectbox(
                        f"Max count for {cls}",
                        options=["None"] + [str(x) for x in st.session_state.count_options[cls]]
                    )

        if st.button("Search Images", type="primary") and st.session_state.search_params["selected_classes"]:
            results = []
            search_params = st.session_state.search_params

            for item in st.session_state.metadata:
                class_matches = {}

                for cls in search_params["selected_classes"]:
                    class_detections = [d for d in item.get('detections', []) if d['class'] == cls]
                    class_count = len(class_detections)
                    threshold = search_params["thresholds"].get(cls, "None")

                    if threshold == "None":
                        class_matches[cls] = (class_count >= 1)
                    else:
                        class_matches[cls] = (1 <= class_count <= int(threshold))

                if search_params["search_mode"] == "Any of selected classes (OR)":
                    matches = any(class_matches.values())
                else:
                    matches = all(class_matches.values())

                if matches:
                    results.append(item)

            st.session_state.search_results = results

# Display Results
if st.session_state.search_results:
    results = st.session_state.search_results
    search_params = st.session_state.search_params

    st.subheader(f"📷 Results: {len(results)} matching images")

    with st.expander("Display Options", expanded=True):
        cols = st.columns(3)
        with cols[0]:
            st.session_state.show_boxes = st.checkbox("Show bounding boxes", value=st.session_state.show_boxes)
        with cols[1]:
            st.session_state.grid_columns = st.slider("Grid columns", min_value=2, max_value=6, value=st.session_state.grid_columns)
        with cols[2]:
            st.session_state.highlight_matches = st.checkbox("Highlight matching classes", value=st.session_state.highlight_matches)

    grid_cols = st.columns(st.session_state.grid_columns)
    col_index = 0

    for result in results:
        with grid_cols[col_index]:
            try:
                img_path = result["image_path"]
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    draw = ImageDraw.Draw(img)

                    if st.session_state.show_boxes:
                        try:
                            font = ImageFont.truetype("arial.ttf", 12)
                        except:
                            font = ImageFont.load_default()

                        for det in result.get('detections', []):
                            cls = det['class']
                            bbox = det['bbox']

                            if cls in search_params["selected_classes"]:
                                color = "#30C938"
                                thickness = 3
                            elif not st.session_state.highlight_matches:
                                color = "#666666"
                                thickness = 1
                            else:
                                continue

                            draw.rectangle(bbox, outline=color, width=thickness)

                            if cls in search_params["selected_classes"] or not st.session_state.highlight_matches:
                                label = f"{cls} {det['confidence']:.2f}"
                                text_bbox = draw.textbbox((0, 0), label, font=font)
                                text_width = text_bbox[2] - text_bbox[0]
                                text_height = text_bbox[3] - text_bbox[1]

                                draw.rectangle([bbox[0], bbox[1], bbox[0] + text_width + 8, bbox[1] + text_height + 4], fill=color)
                                draw.text((bbox[0] + 4, bbox[1] + 2), label, fill="white", font=font)

                    meta_items = [f"{k}: {v}" for k, v in result.get('class_counts', {}).items() if k in search_params["selected_classes"]]

                    st.markdown(f"""
                    <div class="image-card">
                        <div class="image-container"><img src="data:image/png;base64,{img_to_base64(img)}"></div>
                        <div class="meta-overlay">
                            <strong>{Path(img_path).name}</strong><br>
                            {", ".join(meta_items) if meta_items else "No matches"}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning(f"Image not found at {img_path}")

            except Exception as e:
                st.error(f"Error displaying image: {str(e)}")

        col_index = (col_index + 1) % st.session_state.grid_columns

    with st.expander("Export Options"):
        st.download_button("Download Results (JSON)", data=json.dumps(results, indent=2), file_name="search_results.json", mime="application/json")