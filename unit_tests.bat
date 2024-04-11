@echo off
setlocal

REM Init options
set verbose=0
set open_browser=n
set ask_open_browser=1

REM Parse command line arguments
:parse
if "%~1"=="" goto :continue
if /i "%~1"=="-v" (
    set verbose=1
    shift
    goto :parse
)
if /i "%~1"=="-o" (
    set ask_open_browser=0
    set open_browser=y
    shift
    goto :parse
)
if /i "%~1"=="-a" (
    set ask_open_browser=0
    set open_browser=%~2
    shift
    shift
    goto :parse
)
echo Invalid argument: %~1
exit /b 1

:continue

REM Run the tests
if "%verbose%"=="1" (
    coverage run -m pytest -v
) else (
    coverage run -m pytest
)

REM Generate the coverage report
coverage report
coverage html
coverage xml

REM Open htmlcov/index.html in default browser after confirming user wants to do so
REM by default, assume no
if "%ask_open_browser%"=="1" (
    set /p open_browser=Open coverage report in default browser? (y/n) 
)
if /i "%open_browser%"=="y" (
    start htmlcov\index.html
)