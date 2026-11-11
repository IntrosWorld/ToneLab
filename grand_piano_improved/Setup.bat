@echo off
setlocal
cd /d "%~dp0"
title Grand Piano - Setup
set "PIANO_PY="
set "PIANO_ARGS="
for %%P in (python.exe) do if not "%%~$PATH:P"=="" set "PIANO_PY=%%~$PATH:P"
if not defined PIANO_PY for %%P in (py.exe) do if not "%%~$PATH:P"=="" (
  set "PIANO_PY=%%~$PATH:P"
  set "PIANO_ARGS=-3"
)
if not defined PIANO_PY (
  echo Install Python 3.10 or newer with tkinter, then run Setup.bat again.
  pause
  exit /b 1
)
"%PIANO_PY%" %PIANO_ARGS% -c "import sys, tkinter; assert sys.version_info >= (3, 10), 'Python 3.10 or newer is required'"
if errorlevel 1 goto failed
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo Creating a Python environment inside this project copy...
  "%PIANO_PY%" %PIANO_ARGS% -m venv "%~dp0.venv"
  if errorlevel 1 goto failed
)
echo Installing the piano renderer dependencies...
"%~dp0.venv\Scripts\python.exe" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 goto failed
"%~dp0.venv\Scripts\python.exe" "%~dp0tools\piano_gui.py" --check
if errorlevel 1 goto failed
echo.
echo Setup complete. Double-click Piano.bat to open Grand Piano.
pause
exit /b 0

:failed
echo.
echo Setup did not finish. Review the error above and run Setup.bat again.
pause
exit /b 1
