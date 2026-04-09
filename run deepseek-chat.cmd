@echo off
cd /d "C:\Users\jwmcg\OneDrive\Documents\AI Projects\Claw Code"

powershell -NoExit -ExecutionPolicy Bypass -Command ^
"$env:HOME=$env:USERPROFILE; $env:OPENAI_BASE_URL='https://api.deepseek.com/v1'; $env:OPENAI_API_KEY=[Environment]::GetEnvironmentVariable('DEEPSEEK_API_KEY','User'); Write-Host ('OPENAI_API_KEY present? ' + [bool]$env:OPENAI_API_KEY); .\rust\target\debug\claw.exe --model 'deepseek-chat'; Write-Host ('Exit code: ' + $LASTEXITCODE)"