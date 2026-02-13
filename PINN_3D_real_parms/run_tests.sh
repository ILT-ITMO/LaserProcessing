#!/bin/bash

# ANSI Color Codes
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "🚀 ${GREEN}Starting PINN Laser Model Test Suite...${NC}"

# Check for pytest
if ! command -v pytest &> /dev/null
then
    echo -e "${RED}Error: pytest not found. Installing...${NC}"
    pip install pytest
fi

# Run tests
pytest tests/

if [ $? -eq 0 ]; then
    echo -e "✅ ${GREEN}All tests passed! project is stable.${NC}"
else
    echo -e "❌ ${RED}Some tests failed. Please check the logs above.${NC}"
    exit 1
fi
