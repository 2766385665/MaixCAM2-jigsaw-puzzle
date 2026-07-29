@echo off
cd /d "%~dp0"
set "PLAN=%~1"
if "%PLAN%"=="" set "PLAN=demo_motion_plan.json"
"D:\label\python\python.exe" pc_motion_simulator.py "%PLAN%"
if errorlevel 1 pause
