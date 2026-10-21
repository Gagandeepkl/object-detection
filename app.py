"""
VisionAI - Real-Time Object Detection Using YOLO and Computer Vision.
Main Streamlit Application.
"""

from io import BytesIO
from pathlib import Path
import time
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from config import (
    APP_ICON,
    APP_SUBTITLE,
    APP_TITLE,
    AVAILABLE_MODELS,
    CONFIDENCE_STEP,
    DEFAULT_CAMERA_INDEX,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_PATH,
    MAX_CONFIDENCE_THRESHOLD,
    MIN_CONFIDENCE_THRESHOLD,
    OUTPUTS_DIR,
    SAMPLE_IMAGES_DIR,
    SUPPORTED_IMAGE_EXTENSIONS,
)
from src.detector import (
    DetectionResult,
    InvalidThresholdError,
    ModelLoadError,
    YOLODetector,
)
from src.image_utils import (
    ImageDecodeError,
    InvalidImageError,
    annotate_image,
    load_image_from_bytes,
    load_image_from_path,
    save_annotated_image,
    validate_uploaded_file,
)
from src.webcam import WebcamAccessError, WebcamFrameError, WebcamManager

# -----------------------------------------------------------------------------
# Streamlit Page Setup & Custom Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title=f"{APP_TITLE} | YOLOv8",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern visual polish
st.markdown(
    """
    <style>
    /* Metric Cards Styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        margin-bottom: 12px;
        backdrop-filter: blur(8px);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.85rem;
        color: #38bdf8;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 2px;
    }

    /* Tag Pills for Classes */
    .class-pill {
        display: inline-block;
        padding: 4px 12px;
        margin: 4px 6px 4px 0;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        background: rgba(56, 189, 248, 0.15);
        border: 1px solid rgba(56, 189, 248, 0.4);
        color: #38bdf8;
    }

    /* Info Callout */
    .custom-info-box {
        background: rgba(14, 165, 233, 0.08);
        border-left: 4px solid #0284c7;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 12px 0;
        font-size: 0.92rem;
    }

    /* Instruction Badge */
    .step-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        background-color: #0284c7;
        color: white;
        font-size: 0.75rem;
        font-weight: bold;
        margin-right: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Model Caching
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="⏳ Loading YOLO Pretrained Weights...")
def get_yolo_detector(model_path: str) -> Optional[YOLODetector]:
    """
    Cached loader for YOLO detector model.
    Guarantees weights are loaded once in memory and reused across runs.
    """
    try:
        return YOLODetector(model_path)
    except ModelLoadError as exc:
        st.error(f"⚠️ **Model Loading Error:** {str(exc)}")
        return None
    except Exception as exc:
        st.error(f"⚠️ **Unexpected Error Loading Model:** {str(exc)}")
        return None


# -----------------------------------------------------------------------------
# Sidebar Configuration & Navigation
# -----------------------------------------------------------------------------
def render_sidebar():
    """Render sidebar navigation, model selection, and detection controls."""
    with st.sidebar:
        st.markdown(f"## {APP_ICON} VisionAI")
        st.caption("Real-Time Object Detection Pipeline")
        st.markdown("---")

        # Navigation
        st.markdown("### 🧭 Navigation")
        page = st.radio(
            "Select Module",
            options=["🖼️ Image Detection", "📹 Live Webcam", "ℹ️ Model Info & Guide"],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("### ⚙️ Detection Settings")

        # Model Selector
        selected_model_name = st.selectbox(
            "YOLO Model Variant",
            options=AVAILABLE_MODELS,
            index=AVAILABLE_MODELS.index(DEFAULT_MODEL_NAME) if DEFAULT_MODEL_NAME in AVAILABLE_MODELS else 0,
            help="Choose the model weights. Nano (yolov8n) is recommended for fast real-time inference.",
        )

        # Resolve model path (fallback to name if not local file)
        local_model_file = Path(selected_model_name)
        model_path = str(local_model_file) if local_model_file.exists() else selected_model_name

        # Confidence Threshold Slider
        st.markdown("**Confidence Threshold**")
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=float(MIN_CONFIDENCE_THRESHOLD),
            max_value=float(MAX_CONFIDENCE_THRESHOLD),
            value=float(DEFAULT_CONFIDENCE_THRESHOLD),
            step=float(CONFIDENCE_STEP),
            help="Minimum confidence score required for an object detection to be accepted.",
            label_visibility="collapsed",
        )

        # Dynamic indicator explaining threshold behavior
        if conf_threshold >= 0.85:
            st.caption("🔒 **Strict Threshold (0.85+)**: Filters low-confidence boxes. Eliminates false positives, but may miss partially obscured objects.")
        elif conf_threshold <= 0.15:
            st.caption("🔓 **Permissive Threshold (<0.15)**: Detects fainter objects, but might include false positives or background artifacts.")
        else:
            st.caption(f"⚖️ **Balanced Mode ({conf_threshold:.2f})**: Recommended default for standard photos and camera feeds.")

        st.markdown("---")

        # First-Time User Quick Guide
        with st.expander("💡 Quick Instructions"):
            st.markdown(
                """
                1. **Image Mode**: Upload any photo (JPG, PNG) or test bundled benchmark images.
                2. **Webcam Mode**: Click *Start Camera* to stream live video with bounding boxes.
                3. **Tune Slider**: Increase confidence for fewer false alarms; decrease for faint objects.
                4. **Download**: Save the annotated image with one click.
                """
            )

        st.caption("Powered by Ultralytics YOLOv8 & OpenCV")

    return page, model_path, conf_threshold


# -----------------------------------------------------------------------------
# Module 1: Image Detection Page
# -----------------------------------------------------------------------------
def render_image_detection_page(detector: Optional[YOLODetector], conf_threshold: float):
    """Render image upload, inference execution, summary statistics, and visual annotations."""
    st.markdown(f"## 🖼️ Image Object Detection")
    st.markdown("Upload a photo or choose from bundled sample benchmark images to detect objects.")

    if detector is None:
        st.error("❌ The YOLO model is currently unavailable. Please check the model configuration in the sidebar.")
        return

    # Image source tabs
    tab_upload, tab_sample = st.tabs(["📁 Upload Your Image", "🌟 Test Included Sample Images"])

    image_to_process: Optional[np.ndarray] = None
    image_title = "Uploaded Image"

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Choose a JPG, JPEG, or PNG image",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG (Max 20MB)",
        )

        if uploaded_file is not None:
            try:
                validate_uploaded_file(uploaded_file)
                image_bytes = uploaded_file.read()
                image_to_process = load_image_from_bytes(image_bytes)
                image_title = uploaded_file.name
            except InvalidImageError as e:
                st.warning(f"⚠️ **Invalid Image:** {str(e)}")
            except ImageDecodeError as e:
                st.error(f"❌ **Image Decoding Error:** {str(e)}")
            except Exception as e:
                st.error(f"❌ **Error reading file:** {str(e)}")

    with tab_sample:
        st.markdown("Select a real benchmark image included in the repository:")
        sample_options = {
            "Dog, Bicycle & Vehicles (Iconic Darknet)": "dog.jpg",
            "Zinedine Zidane & Suits (Portrait)": "person.jpg",
            "City Bus & Pedestrians (Urban Scene)": "bus.jpg",
            "Blank Gradient Texture (Zero Objects Benchmark)": "blank_texture.jpg",
        }
        chosen_sample_name = st.selectbox("Select Sample Image", options=list(sample_options.keys()))
        selected_file_name = sample_options[chosen_sample_name]
        sample_path = SAMPLE_IMAGES_DIR / selected_file_name

        if st.button("🚀 Load Sample Image", type="secondary"):
            try:
                image_to_process = load_image_from_path(sample_path)
                image_title = f"Sample: {selected_file_name}"
                st.session_state["active_sample"] = selected_file_name
            except Exception as e:
                st.error(f"Failed to load sample image: {str(e)}")
        elif st.session_state.get("active_sample") and uploaded_file is None:
            # Retain sample image across slider adjustments
            try:
                sample_path = SAMPLE_IMAGES_DIR / st.session_state["active_sample"]
                image_to_process = load_image_from_path(sample_path)
                image_title = f"Sample: {st.session_state['active_sample']}"
            except Exception:
                pass

    # Process and display detections if an image is loaded
    if image_to_process is not None:
        st.markdown("---")

        # Run inference
        try:
            detection_result = detector.detect(
                image_input=image_to_process,
                conf_threshold=conf_threshold,
            )
        except InvalidThresholdError as e:
            st.error(f"⚠️ Invalid Confidence Threshold: {str(e)}")
            return
        except Exception as e:
            st.error(f"❌ Object detection inference failed: {str(e)}")
            return

        # Render annotated image
        annotated_img = annotate_image(image_to_process, detection_result.detections)

        # Overview Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Total Objects</div>
                    <div class="metric-value">{detection_result.total_count}</div>
                    <div class="metric-sub">qualifying detections</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Unique Classes</div>
                    <div class="metric-value">{len(detection_result.class_counts)}</div>
                    <div class="metric-sub">distinct categories</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Inference Time</div>
                    <div class="metric-value">{detection_result.inference_time_ms} ms</div>
                    <div class="metric-sub">YOLOv8 forward pass</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Active Threshold</div>
                    <div class="metric-value">{conf_threshold:.2f}</div>
                    <div class="metric-sub">minimum confidence</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Side-by-Side Images (Original vs Annotated)
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.markdown(f"**Original Image:** `{image_title}`")
            st.image(image_to_process, width="stretch")

        with img_col2:
            st.markdown("**Annotated Detection Output:**")
            st.image(annotated_img, width="stretch")

            # Download Button
            pil_annotated = Image.fromarray(annotated_img)
            buf = BytesIO()
            pil_annotated.save(buf, format="JPEG", quality=95)
            byte_im = buf.getvalue()

            st.download_button(
                label="📥 Download Annotated Image",
                data=byte_im,
                file_name=f"detected_{Path(image_title).stem}_{int(time.time())}.jpg",
                mime="image/jpeg",
                width="stretch",
            )

        # Detection Summary Section
        st.markdown("### 📊 Detection Summary")

        if not detection_result.has_detections:
            st.info(
                f"ℹ️ **No qualifying objects detected.** No objects met the configured confidence threshold of "
                f"**{conf_threshold:.2f}**. Try lowering the confidence slider in the sidebar to reveal fainter detections."
            )
        else:
            # Breakdown by class
            st.markdown("**Detected Categories:**")
            pills_html = "".join(
                [
                    f'<span class="class-pill">🏷️ {cls_name}: <b>{count}</b></span>'
                    for cls_name, count in sorted(detection_result.class_counts.items(), key=lambda x: -x[1])
                ]
            )
            st.markdown(pills_html, unsafe_allow_html=True)

            # Detailed Detections Table
            st.markdown("#### Detailed Bounding Box Data")
            table_records = []
            for idx, det in enumerate(detection_result.detections, 1):
                xmin, ymin, xmax, ymax = det.bbox
                table_records.append(
                    {
                        "#": idx,
                        "Class": det.class_name,
                        "Confidence": f"{det.confidence * 100:.1f}%",
                        "Xmin": xmin,
                        "Ymin": ymin,
                        "Xmax": xmax,
                        "Ymax": ymax,
                        "Box Width": xmax - xmin,
                        "Box Height": ymax - ymin,
                    }
                )

            df = pd.DataFrame(table_records)
            st.dataframe(df, width="stretch", hide_index=True)


# -----------------------------------------------------------------------------
# Module 2: Live Webcam Detection Page
# -----------------------------------------------------------------------------
def render_webcam_page(detector: Optional[YOLODetector], conf_threshold: float):
    """Render real-time live webcam detection feed and snapshot camera input."""
    st.markdown(f"## 📹 Live Webcam Object Detection")
    st.markdown("Perform continuous real-time YOLO object detection using your connected camera feed.")

    if detector is None:
        st.error("❌ The YOLO model is currently unavailable. Please verify model settings.")
        return

    # Two operational modes: Continuous OpenCV Stream & Browser Snapshot
    cam_mode = st.radio(
        "Choose Camera Mode",
        options=["Continuous Live Stream (OpenCV)", "Webcam Snapshot (Browser Camera)"],
        horizontal=True,
    )

    if cam_mode == "Continuous Live Stream (OpenCV)":
        st.markdown(
            """
            <div class="custom-info-box">
                <b>Continuous Live Stream Mode:</b> Reads successive frames from OpenCV <code>VideoCapture</code>, 
                annotates detected objects in real-time, and calculates live processing FPS.
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 2])
        with col_ctrl1:
            cam_idx = st.number_input("Camera Index", min_value=0, max_value=5, value=DEFAULT_CAMERA_INDEX, step=1)
        with col_ctrl2:
            run_cam = st.checkbox("▶️ Start Webcam Detection", value=False, key="webcam_active")

        # Placeholders for live dynamic updates
        hud_col1, hud_col2, hud_col3 = st.columns(3)
        with hud_col1:
            fps_metric = st.empty()
        with hud_col2:
            count_metric = st.empty()
        with hud_col3:
            class_metric = st.empty()

        frame_display = st.empty()
        status_message = st.empty()

        if run_cam:
            manager = WebcamManager(camera_index=int(cam_idx))
            try:
                status_message.info("🔄 Connecting to camera hardware...")
                manager.open_camera()
                status_message.success("🟢 Camera connected. Streaming live detections (Uncheck 'Start Webcam' to stop).")

                while st.session_state.get("webcam_active", False):
                    try:
                        frame_rgb = manager.read_frame()
                    except WebcamFrameError as e:
                        status_message.warning(f"Frame capture issue: {e}")
                        break

                    # Inference pass
                    result = detector.detect(frame_rgb, conf_threshold=conf_threshold)

                    # Annotate frame
                    annotated_frame = annotate_image(frame_rgb, result.detections)

                    # Update HUD metrics
                    current_fps = manager.get_fps()
                    fps_metric.metric("Processing FPS", f"{current_fps} fps")
                    count_metric.metric("Objects in View", f"{result.total_count}")

                    classes_str = ", ".join(result.class_counts.keys()) if result.class_counts else "None"
                    class_metric.metric("Detected Classes", classes_str)

                    # Render live frame
                    frame_display.image(annotated_frame, channels="RGB", width="stretch")

                    # Yield for UI event loop
                    time.sleep(0.03)

            except WebcamAccessError as e:
                status_message.error(f"❌ **Camera Access Error:**\n\n{str(e)}")
            except Exception as e:
                status_message.error(f"❌ **Unexpected Error during camera stream:** {str(e)}")
            finally:
                # Guaranteed cleanup of hardware resources
                manager.release()
                if not st.session_state.get("webcam_active", False):
                    status_message.info("⏹️ Camera stopped. Hardware resources released cleanly.")
        else:
            frame_display.info("Click **'Start Webcam Detection'** above to activate the camera.")

    else:
        # Browser Snapshot Mode (Works on remote headless deployments or restricted browser sandbox)
        st.markdown(
            """
            <div class="custom-info-box">
                <b>Webcam Snapshot Mode:</b> Utilizes the browser's native camera API. Ideal for cloud servers, 
                Docker containers, or devices where direct OpenCV hardware access is restricted.
            </div>
            """,
            unsafe_allow_html=True,
        )

        picture = st.camera_input("Take a photo using your webcam")

        if picture is not None:
            try:
                bytes_data = picture.getvalue()
                frame_rgb = load_image_from_bytes(bytes_data)

                # Inference
                result = detector.detect(frame_rgb, conf_threshold=conf_threshold)
                annotated_frame = annotate_image(frame_rgb, result.detections)

                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.markdown("**Captured Webcam Frame:**")
                    st.image(frame_rgb, width="stretch")
                with res_col2:
                    st.markdown("**Annotated Detections:**")
                    st.image(annotated_frame, width="stretch")

                st.markdown(f"**Objects Detected:** {result.total_count} | **Inference Time:** {result.inference_time_ms} ms")
                if result.has_detections:
                    st.json({k: v for k, v in result.class_counts.items()})
                else:
                    st.info("No objects detected above the selected confidence threshold.")

            except Exception as e:
                st.error(f"Error processing camera snapshot: {str(e)}")


# -----------------------------------------------------------------------------
# Module 3: Model Info, Architecture & Presentation Guide
# -----------------------------------------------------------------------------
def render_model_info_page():
    """Render comprehensive educational documentation for college viva and project demonstrations."""
    st.markdown(f"## ℹ️ Technical Overview & Demonstration Guide")

    guide_tabs = st.tabs(["🧠 YOLO Architecture", "📊 COCO Classes", "🎓 College Viva Q&A", "⚙️ System Specs"])

    with guide_tabs[0]:
        st.markdown(
            """
            ### Understanding the YOLOv8 Detection Pipeline

            **YOLO (You Only Look Once)** revolutionized computer vision by reframing object detection 
            from a two-stage classification/proposal problem (like Faster R-CNN) into a single unified regression problem.

            #### Key Architectural Highlights:
            1. **CSPDarknet Backbone**:
               - Employs Cross-Stage Partial connections to reduce redundant gradient information while preserving high-resolution visual features.
               - Extracts multi-scale feature maps from coarse to fine spatial resolutions.
            2. **PAN-FPN Neck (Path Aggregation Network)**:
               - Merges semantic context from top-down layers with precise spatial localization from bottom-up layers.
               - Enables accurate detection of both small objects (e.g. ties, bottles) and large objects (e.g. buses, airplanes).
            3. **Anchor-Free Decoupled Head**:
               - Unlike older YOLO versions (v3/v4) that relied on predefined anchor box sizes, YOLOv8 predicts bounding box offsets directly from feature centroids.
               - Decouples the **classification branch** from the **bounding box regression branch**, significantly accelerating convergence and mAP.
            4. **Non-Maximum Suppression (NMS)**:
               - Eliminates redundant overlapping bounding boxes for the same object by calculating Intersection over Union (IoU) and keeping the highest-confidence prediction.
            """
        )

    with guide_tabs[1]:
        st.markdown(
            """
            ### Pretrained MS COCO Dataset (80 Classes)

            The default model (`yolov8n.pt`) is pretrained on the **Microsoft COCO (Common Objects in Context)** benchmark dataset containing 80 everyday categories:

            - **People & Accessories**: `person`, `bicycle`, `backpack`, `umbrella`, `handbag`, `tie`, `suitcase`
            - **Vehicles**: `car`, `motorcycle`, `airplane`, `bus`, `train`, `truck`, `boat`
            - **Animals**: `bird`, `cat`, `dog`, `horse`, `sheep`, `cow`, `elephant`, `bear`, `zebra`, `giraffe`
            - **Traffic Elements**: `traffic light`, `fire hydrant`, `stop sign`, `parking meter`, `bench`
            - **Indoor & Everyday Electronics**: `chair`, `couch`, `bed`, `dining table`, `tv`, `laptop`, `mouse`, `keyboard`, `cell phone`
            - **Kitchen & Food**: `bottle`, `wine glass`, `cup`, `fork`, `knife`, `spoon`, `bowl`, `banana`, `apple`, `sandwich`, `orange`, `pizza`
            """
        )

    with guide_tabs[2]:
        st.markdown(
            """
            ### 🎓 Frequently Asked Questions for College Project Viva

            **Q1: What does 'Real-Time' mean in this application?**  
            *A:* Real-time means the processing pipeline (frame capture, YOLO forward pass, NMS, annotation, and display) executes fast enough (typically >15-30 frames per second on CPU/GPU) to process video frames without perceptible lag.

            **Q2: What is the effect of changing the confidence threshold from 0.25 to 0.90?**  
            *A:* The confidence threshold acts as a filter on the model's Softmax/Sigmoid probability score:
            - **At 0.25 (Permissive)**: More candidate bounding boxes qualify. Useful for small or occluded objects, but increases risk of false positives.
            - **At 0.90 (Strict)**: Only predictions where the model is at least 90% certain are displayed. This virtually eliminates false positives, but valid objects with partial visibility may be omitted.

            **Q3: Why use YOLOv8 Nano (`yolov8n.pt`) instead of YOLOv8 Extra-Large (`yolov8x.pt`)?**  
            *A:* YOLOv8n has only ~3.2 million parameters and a weight file of ~6.5MB, making it ideal for edge devices and CPU real-time inference (30-60ms/frame), whereas YOLOv8x requires dedicated GPU resources.

            **Q4: How does OpenCV handle video capture and release?**  
            *A:* OpenCV's `cv2.VideoCapture` interfaces with low-level OS drivers (DirectShow on Windows or V4L2 on Linux). Proper release via `cap.release()` is crucial to free OS camera handles and prevent device lockouts.
            """
        )

    with guide_tabs[3]:
        st.markdown("### 🖥️ Application & Environment Runtime Information")
        st.markdown(
            f"""
            - **Python Version**: `3.10+`
            - **Object Detection Engine**: Ultralytics YOLOv8
            - **Computer Vision Framework**: OpenCV (`cv2`)
            - **User Interface**: Streamlit
            - **Pretrained Weights**: `{DEFAULT_MODEL_NAME}` (Cached in memory)
            - **Output Directory**: `{OUTPUTS_DIR}`
            - **Bundled Benchmark Images**: `{SAMPLE_IMAGES_DIR}`
            """
        )


# -----------------------------------------------------------------------------
# Main Application Entry Point
# -----------------------------------------------------------------------------
def main():
    """Main application orchestrator."""
    page, model_path, conf_threshold = render_sidebar()

    # Load shared cached YOLO detector
    detector = get_yolo_detector(model_path)

    if page == "🖼️ Image Detection":
        render_image_detection_page(detector, conf_threshold)
    elif page == "📹 Live Webcam":
        render_webcam_page(detector, conf_threshold)
    elif page == "ℹ️ Model Info & Guide":
        render_model_info_page()


if __name__ == "__main__":
    main()
