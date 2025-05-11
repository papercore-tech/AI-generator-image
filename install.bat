@echo off

REM Создание виртуального окружения
python -m venv .venv

REM Активация окружения
call .venv\Scripts\activate.bat

REM Установка зависимостей
pip install --upgrade pip
pip install -r requirements.txt

pause
