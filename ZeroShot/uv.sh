#!/bin/bash

# Create virtual environment
uv venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies using uv
uv pip install -r requirements.txt

echo "Virtual environment created and dependencies installed."