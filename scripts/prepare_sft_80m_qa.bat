@echo off
setlocal

set PROJECT_ROOT=%~dp0..
cd /d "%PROJECT_ROOT%"

python -B scripts\prepare_public_sft_data.py --target-size 150000 --skip-failed-sources --force %*

endlocal
