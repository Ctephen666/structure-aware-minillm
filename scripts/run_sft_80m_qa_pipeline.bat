@echo off
setlocal

set PROJECT_ROOT=%~dp0..
cd /d "%PROJECT_ROOT%"

call scripts\prepare_sft_80m_qa.bat
if errorlevel 1 exit /b %errorlevel%

call scripts\run_sft_80m_qa_5h.bat
if errorlevel 1 exit /b %errorlevel%

endlocal
