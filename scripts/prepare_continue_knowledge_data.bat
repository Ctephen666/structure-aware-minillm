@echo off
setlocal

set PROJECT_ROOT=%~dp0..
cd /d "%PROJECT_ROOT%"

python -B scripts\download_continue_knowledge_data.py %*
if errorlevel 1 exit /b %errorlevel%

python -B scripts\generate_continue_structure_data.py
if errorlevel 1 exit /b %errorlevel%

python -B scripts\build_continue_knowledge_mix.py --drop-empty-sources --force
if errorlevel 1 exit /b %errorlevel%

python -B scripts\count_tokens.py --config configs\struct_continue_80m_knowledge.yaml
if errorlevel 1 exit /b %errorlevel%

endlocal
