@echo off
setlocal
cd /d "%~dp0"
title Grand Piano - MIDI to WAV

REM Use this copy's isolated environment when Setup.bat has created it.
set "PIANO_PY="
set "PIANO_ARGS="
if exist "%~dp0.venv\Scripts\python.exe" set "PIANO_PY=%~dp0.venv\Scripts\python.exe"
if not defined PIANO_PY for %%P in (python.exe) do if not "%%~$PATH:P"=="" set "PIANO_PY=%%~$PATH:P"
if not defined PIANO_PY for %%P in (py.exe) do if not "%%~$PATH:P"=="" (
  set "PIANO_PY=%%~$PATH:P"
  set "PIANO_ARGS=-3"
)
if not defined PIANO_PY (
  echo Python was not found. Install Python 3.10 or newer with tkinter.
  echo Then double-click Setup.bat in this folder.
  pause
  exit /b 1
)

REM Preflight in this console so dependency failures cannot disappear.
"%PIANO_PY%" %PIANO_ARGS% "%~dp0tools\piano_gui.py" --check
if errorlevel 1 (
  echo.
  echo Double-click Setup.bat to install this app's dependencies, then try again.
  pause
  exit /b 1
)
if /i "%~1"=="--check" exit /b 0
"%PIANO_PY%" %PIANO_ARGS% "%~dp0tools\piano_gui.py" --launch
if errorlevel 1 (
  echo.
  echo The app could not start. Details are in out\logs\piano_gui.log.
  pause
  exit /b 1
)
exit /b 0
