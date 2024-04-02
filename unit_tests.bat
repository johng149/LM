@echo off

REM Run the tests
coverage run -m pytest

REM Generate the coverage report
coverage report
coverage html
coverage xml

REM Open htmlcov/index.html in default browser after confirming user wants to do so
REM by default, assume no
set open_browser=n
set /p open_browser=Open coverage report in default browser? (y/n)
if /i "%open_browser%" EQU "y" start htmlcov/index.html