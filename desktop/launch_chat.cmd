@echo off
title Research RAG Chat
REM Runs from wherever the repo was downloaded (was a hard-coded placeholder path).
cd /d "%~dp0.."
if exist .venv\Scripts\python.exe (set PY=.venv\Scripts\python.exe) else (set PY=python)
%PY% rag\chat.py
echo.
pause
