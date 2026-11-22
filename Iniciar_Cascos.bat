@echo off
setlocal
cd /d "%~dp0"
set "PYTHON_EXE=python"
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
echo =========================================
echo   APARTADO 1 - CONTROL VIAL Y CASCOS
echo =========================================
"%PYTHON_EXE%" iniciar_cascos.py %*
set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo Ventana del iniciador vial finalizada.
pause
exit /b %EXIT_CODE%
