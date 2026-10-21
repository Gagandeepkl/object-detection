@echo off
cd /d "C:\Users\HP\Downloads\object"
call "venv\Scripts\activate.bat"
python -m streamlit run app.py
pause