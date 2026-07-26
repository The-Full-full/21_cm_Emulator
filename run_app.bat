@echo off
echo ==============================================
echo      Starting 21cm Emulator Initialization
echo ==============================================
echo.
echo Checking and installing required packages...
pip install -r requirements.txt

echo.
echo Starting the application...
streamlit run app.py

pause
