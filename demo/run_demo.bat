@echo off
rem Double-click to start the demo. First run creates .venv and installs packages.
setlocal
cd /d "%~dp0"

where py >nul 2>nul && (set "PY=py -3") || (set "PY=python")

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Tao moi truong Python rieng cho demo...
    %PY% -m venv .venv || goto :error
    echo [2/3] Cai thu vien, chi lan dau, mat vai phut...
    ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q -r requirements.txt || goto :error
)

if not exist "artifacts\summary.json" (
    echo Dang danh gia mo hinh, khoang 1 phut...
    ".venv\Scripts\python.exe" build_artifacts.py || goto :error
)

echo [3/3] Demo se mo tai http://localhost:8501  (dong cua so nay de tat)
rem open the browser ourselves a few seconds after the server starts
if not defined DEMO_NO_BROWSER start "" /b cmd /c "timeout /t 5 /nobreak >nul & start http://localhost:8501"
".venv\Scripts\python.exe" -m streamlit run app.py --server.headless true
goto :eof

:error
echo.
echo Co loi xay ra. Kiem tra may da cai Python 3.12 va co ket noi Internet.
pause
exit /b 1
