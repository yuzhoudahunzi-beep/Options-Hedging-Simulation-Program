@echo off
title Option Hedge Trainer v2.0
echo.
echo  ========================================
echo   Option Hedge Trainer v2.0
echo  ========================================
echo.

REM Find Python — prefer known install locations; skip the WindowsApps Store stub.
set "PYTHON_EXE="

if not defined PYTHON_EXE if exist "C:\Users\a1320\AppData\Local\Programs\Python\Python314\python.exe" set "PYTHON_EXE=C:\Users\a1320\AppData\Local\Programs\Python\Python314\python.exe"
if not defined PYTHON_EXE if exist "C:\Python314\python.exe" set "PYTHON_EXE=C:\Python314\python.exe"
if not defined PYTHON_EXE if exist "C:\Python313\python.exe" set "PYTHON_EXE=C:\Python313\python.exe"
if not defined PYTHON_EXE if exist "C:\Python312\python.exe" set "PYTHON_EXE=C:\Python312\python.exe"

REM Fallback: `where python`, but reject the WindowsApps Store stub (WindowsApps\python.exe
REM is a redirector that exits with code 49 instead of running Python).
if not defined PYTHON_EXE (
    for /f "delims=" %%i in ('where python 2^>nul') do (
        if not defined PYTHON_EXE (
            echo %%i | findstr /I "\\WindowsApps\\" >nul
            if errorlevel 1 set "PYTHON_EXE=%%i"
        )
    )
)

if not defined PYTHON_EXE (
    echo [ERROR] Python not found!
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

echo Python: %PYTHON_EXE%
"%PYTHON_EXE%" --version
echo.

cd /d "%~dp0"
"%PYTHON_EXE%" run.py

pause
