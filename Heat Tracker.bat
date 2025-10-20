@echo off
:: ============================================
:: Streamlit Heat Tracker - Local Launcher (Anaconda env)
:: ============================================

echo Starting Streamlit Heat Tracker...
echo ---------------------------------------------

:: Go to this folder
cd /d "%~dp0"

:: Run Streamlit using your Anaconda environment
"C:\Users\MateuszSz\anaconda3\envs\DATA\python.exe" -m streamlit run streamlit_heat_tracker.py --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false

echo ---------------------------------------------
echo Streamlit app stopped.
pause
