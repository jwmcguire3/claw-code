@echo off
cd /d "C:\Users\jwmcg\OneDrive\Documents\AI Projects\Claw Code"

set "HOME=%USERPROFILE%"
set "OPENAI_API_KEY=%OPENAI_API_KEY%"

powershell -NoExit -ExecutionPolicy Bypass -Command ".\rust\target\debug\claw.exe --model 'openai/gpt-4.1-mini'"