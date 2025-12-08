@echo off
call venv\Scripts\activate
python main.py serve --port 8000 --reload
