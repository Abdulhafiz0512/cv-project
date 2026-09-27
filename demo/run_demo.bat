@echo off
REM Run the Junction Watch live demo on this Windows machine.
REM First run creates .venv-demo and installs dependencies (needs internet, ~2 GB).
REM   run_demo.bat            -> public https://*.gradio.live link + http://<this-pc>:7860
REM   run_demo.bat --local    -> this machine only
setlocal
cd /d "%~dp0\.."
set PY=py -3.12
%PY% --version >nul 2>&1 || set PY=python
if not exist .venv-demo\Scripts\python.exe (
  echo Creating .venv-demo ...
  %PY% -m venv .venv-demo || goto :fail
  .venv-demo\Scripts\python.exe -m pip install --upgrade pip || goto :fail
  .venv-demo\Scripts\python.exe -m pip install -r demo\requirements.txt || goto :fail
)
if "%1"=="--local" (
  .venv-demo\Scripts\python.exe demo\app.py
) else (
  .venv-demo\Scripts\python.exe demo\app.py --host 0.0.0.0 --share
)
goto :eof
:fail
echo Setup failed. Install Python 3.10-3.12 from python.org and run this file again.
exit /b 1
