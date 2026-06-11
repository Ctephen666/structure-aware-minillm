@echo off
setlocal

set PROJECT_ROOT=%~dp0..
cd /d "%PROJECT_ROOT%"

if not exist checkpoints mkdir checkpoints
if not exist logs mkdir logs

python -B train\train_struct.py --config configs\struct_pretrain_80m.yaml --device cuda %*

endlocal
