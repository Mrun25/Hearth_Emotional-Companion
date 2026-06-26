@echo off
REM ============================================================
REM  startup_runner.bat -- Fumii Eval Pipeline Boot Launcher
REM  Place this file in: %APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\
REM  It runs automatically every time Windows starts.
REM ============================================================

REM Change to the project root directory
cd /d "D:\Fumii_LLM\fumii-finetune"

REM Activate the virtual environment
call ".venv\Scripts\activate.bat"

REM Run the evaluation pipeline — output goes to the log file
REM (eval_runner.py itself appends to fumii_eval_log.txt)
python pipeline\eval_runner.py >> pipeline\startup_boot.log 2>&1

REM Deactivate the virtual environment
deactivate
