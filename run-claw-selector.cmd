@echo off
cd /d "%~dp0"
powershell -NoExit -ExecutionPolicy Bypass -File ".\launch-claw-selector.ps1"
pause
