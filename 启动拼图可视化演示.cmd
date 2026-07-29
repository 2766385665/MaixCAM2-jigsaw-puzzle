@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
D:\label\python\python.exe puzzle_visual_demo.py
if errorlevel 1 pause
