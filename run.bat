@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install -q -e ".[dev]"
python -m streamlit run app.py --browser.gatherUsageStats=false
