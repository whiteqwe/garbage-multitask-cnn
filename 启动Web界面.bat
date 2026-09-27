@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ========================================================
echo   慧眼 — 智能垃圾分类与回收建议系统
echo   正在启动 Gradio Web 界面，请稍候...
echo ========================================================
echo.
echo 本地访问地址: http://127.0.0.1:7860
echo.
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" app.py
) else (
    python app.py
)
pause
