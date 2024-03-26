#!/bin/bash

# Run the tests
coverage run -m pytest

# Generate the coverage report
coverage report
coverage html
coverage xml

# Open htmlcov/index.html in default browser after confirming user wants to do so
# by default, assume no
open_browser="n"
read -p "Do you want to open the coverage report in your default browser? (y/N) " open_browser
if [ "$open_browser" == "y" ]; then
    open htmlcov/index.html
fi