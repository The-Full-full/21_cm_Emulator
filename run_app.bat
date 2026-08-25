@echo off
echo ==============================================
echo      Starting 21cm Emulator Initialization
echo ==============================================
echo.

if exist .venv_310\Scripts\activate.bat (
    echo [1/3] Activating virtual environment .venv_310...
    call .venv_310\Scripts\activate.bat
) else if exist .venv\Scripts\activate.bat (
    echo [1/3] Activating virtual environment .venv...
    call .venv\Scripts\activate.bat
) else (
    echo [1/3] No virtual environment found. Using global python...
)

echo.
echo [2/3] Checking and installing required packages...
pip install -r requirements.txt

echo.
echo [3/3] Starting the application...
streamlit run app.py

pause
