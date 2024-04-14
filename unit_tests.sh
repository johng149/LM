#!/bin/bash

# Initialize options
verbose=0
open_browser="n"
ask_open_browser=1

# Parse command line arguments
while getopts "vba:" opt; do
    case ${opt} in
        v)
            verbose=1
            ;;
        b)
            ask_open_browser=0
            open_browser="y"
            ;;
        a)
            ask_open_browser=0
            open_browser="$OPTARG"
            ;;
        \?)
            echo "Usage: unit_tests.sh [-v] [-b] [-a y|n]"
            echo "Options:"
            echo "  -v: Verbose output"
            echo "  -b: Open coverage report in default browser"
            echo "  -a y|n: Automatically open coverage report in default browser"
            exit 1
            ;;
    esac
done

# Run the tests
if [ "$verbose" -eq 1 ]; then
    coverage run -m pytest -v
else
    coverage run -m pytest
fi

# Generate the coverage report
coverage report
coverage html
coverage xml

# Open htmlcov/index.html in default browser after confirming user wants to do so
if [ "$ask_open_browser" -eq 1 ]; then
    read -p "Do you want to open the coverage report in your default browser? (y/N) " open_browser
fi
if [ "$open_browser" == "y" ]; then
    open htmlcov/index.html
fi