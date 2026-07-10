@echo off
REM AI MLS Simulation Hospital launcher (Windows)
REM Pinned: streamlit>=1.30, opencv-contrib-python>=4.7 (for ArucoDetector)
pip install "streamlit>=1.30" "opencv-contrib-python>=4.7" numpy scipy pillow plotly pydicom pandas -q
streamlit run hospital_portal.py
pause
