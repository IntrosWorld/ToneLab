@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%~dp0visual\LaunchStyles.ps1" -Style BlueSmoke
if errorlevel 1 (
  echo The piano player could not start. See visual\README.md for setup details.
  pause
  exit /b 1
)
