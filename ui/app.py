
import subprocess
import sys
from pathlib import Path

import streamlit as st


# -------------------------------
# PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="AI Driver Safety System",
    page_icon="🚗",
    layout="centered"
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
MAIN_FILE = PROJECT_DIR / "main.py"


# -------------------------------
# SESSION STATE
# -------------------------------

if "camera_process" not in st.session_state:
    st.session_state.camera_process = None


# -------------------------------
# HELPER FUNCTIONS
# -------------------------------

def start_camera():
    process = st.session_state.camera_process

    if process is not None and process.poll() is None:
        return

    if not MAIN_FILE.exists():
        st.error(f"main.py not found at: {MAIN_FILE}")
        return

    try:
        st.session_state.camera_process = subprocess.Popen(
            [sys.executable, str(MAIN_FILE)],
            cwd=str(PROJECT_DIR),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except Exception as error:
        st.error(f"Could not start camera: {error}")


def stop_camera():
    process = st.session_state.camera_process

    if process is not None and process.poll() is None:
        process.terminate()

        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)

    st.session_state.camera_process = None


def camera_is_running():
    process = st.session_state.camera_process
    return process is not None and process.poll() is None


# -------------------------------
# DASHBOARD HEADER
# -------------------------------

st.title("🚗 AI Driver Safety System")

st.write(
    "Monitor driver safety using computer vision "
    "and AI-based detection."
)

st.divider()


# -------------------------------
# SYSTEM STATUS
# -------------------------------

st.subheader("System Status")

if camera_is_running():
    st.success("🟢 Camera system is running")
else:
    st.info("⚪ Camera system is stopped")

st.caption(
    "The camera feed opens in a separate OpenCV window."
)

st.divider()


# -------------------------------
# CAMERA CONTROLS
# -------------------------------

st.subheader("Camera Controls")

col1, col2 = st.columns(2)

with col1:
    if st.button(
        "▶ Start Camera",
        use_container_width=True,
        type="primary",
        disabled=camera_is_running()
    ):
        start_camera()
        st.rerun()

with col2:
    if st.button(
        "⏹ Stop Camera",
        use_container_width=True,
        disabled=not camera_is_running()
    ):
        stop_camera()
        st.rerun()


# -------------------------------
# DETECTION INFORMATION
# -------------------------------

st.divider()

st.subheader("Detection Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("😴 Drowsiness", "Enabled")

with col2:
    st.metric("📱 Phone Detection", "Enabled")

with col3:
    st.metric("🥱 Yawning", "Enabled")


# -------------------------------
# FOOTER
# -------------------------------

st.divider()

st.caption(
    "AI Driver Safety System | Student Prototype"
)
