#!/bin/bash
# AI MLS Simulation Hospital launcher (Mac/Linux)
# Pinned: streamlit>=1.30, opencv-contrib-python>=4.7 (for ArucoDetector)
set -e
pip install "streamlit>=1.30" "opencv-contrib-python>=4.7" numpy scipy pillow plotly pydicom pandas -q
streamlit run hospital_portal.py
