# Testing Guide

This document explains how to run the test suite for fishsense-api-sdk.

## Prerequisites

Make sure you have Python 3.12 or higher installed.

## Installation

Install the package with test dependencies:

```bash
pip install -e .
pip install pytest pytest-asyncio pytest-mock pytest-cov
```

Or if using uv:

```bash
uv sync --all-groups
```

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run tests with verbose output
```bash
pytest tests/ -v
```

### Run tests with coverage report
```bash
pytest tests/ --cov=fishsense_api_sdk --cov-report=term-missing
```

### Run tests for a specific module
```bash
# Test only models
pytest tests/models/

# Test only clients
pytest tests/clients/

# Test a specific file
pytest tests/clients/test_fish_client.py
```

### Run tests in quiet mode
```bash
pytest tests/ -q
```

## Test Structure

The test suite is organized as follows:

```
tests/
├── __init__.py
├── models/                    # Model tests
│   ├── __init__.py
│   ├── test_model_base.py    # Base model functionality
│   ├── test_fish.py          # Fish model
│   ├── test_species.py       # Species model
│   ├── test_user.py          # User model
│   └── test_measurement.py   # Measurement model
├── clients/                   # Client tests
│   ├── __init__.py
│   ├── test_client_base.py   # Base client functionality
│   ├── test_fish_client.py   # FishClient
│   └── test_user_client.py   # UserClient
└── test_client.py             # Main Client class tests
```

## Coverage Report

After running tests with coverage, you can view the HTML report:

```bash
# Generate HTML coverage report
pytest tests/ --cov=fishsense_api_sdk --cov-report=html

# Open the report in your browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Current Test Statistics

- **Total Tests**: 62
- **Coverage**: 68%
- **Status**: All passing ✅

## Writing New Tests

When adding new tests, follow these guidelines:

1. **Location**: Place tests in the appropriate directory (`models/` or `clients/`)
2. **Naming**: Test files should start with `test_` and test functions should start with `test_`
3. **Mocking**: Use `pytest-mock` for mocking HTTP requests and external dependencies
4. **Async Tests**: Use `async def` for async test functions - pytest-asyncio will handle them automatically
5. **Assertions**: Use clear, descriptive assertions
6. **Documentation**: Add docstrings to test functions explaining what they test

### Example Test

```python
import asyncio
from unittest.mock import AsyncMock, Mock, patch

async def test_get_fish_by_id(self):
    """Test getting a fish by ID."""
    # Arrange
    semaphore = asyncio.Semaphore(10)
    client = FishClient(
        base_url="http://test.com",
        username="testuser",
        password="testpass",
        timeout=10,
        semaphore=semaphore,
    )
    
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"id": 1, "species_id": 100}
    
    # Act
    with patch.object(client, "_get", new_callable=AsyncMock, return_value=mock_response):
        async with client:
            fish = await client.get(fish_id=1)
    
    # Assert
    assert fish.id == 1
    assert fish.species_id == 100
```

## Continuous Integration

Tests are automatically run on every commit via GitHub Actions. Make sure all tests pass before merging.

## Troubleshooting

### Import Errors
If you get import errors, make sure you've installed the package in editable mode:
```bash
pip install -e .
```

### Async Warnings
If you see warnings about async loops, make sure you're using the latest version of pytest-asyncio:
```bash
pip install --upgrade pytest-asyncio
```
