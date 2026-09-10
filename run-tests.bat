@echo off
TITLE Magika AI - Test Suite

echo [1/2] Running PyTest Suite...
python -m pytest test_main.py -v

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Tests failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [2/2] All tests passed successfully!
pause