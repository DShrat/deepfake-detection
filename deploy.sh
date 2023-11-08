#!/bin/bash

# Install Python packages from requirements.txt
pip install -r requirements.txt

# Start your application server (e.g., using uvicorn)
uvicorn main:app --host 0.0.0.0 --port $PORT
