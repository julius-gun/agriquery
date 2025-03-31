#!/bin/bash

# Start Ollama in the background
ollama serve &

# Wait for Ollama to be ready (check if the API is responding)
# This is a more robust way to ensure Ollama is up before starting main.py
attempts=0
while ! curl -s -o /dev/null -w "%{http_code}" http://localhost:11434/api/show && [[ $attempts -lt 30 ]]; do
  attempts=$((attempts + 1))
  echo "Waiting for Ollama to start (attempt $attempts)..."
  sleep 1
done

if [[ $attempts -eq 30 ]]; then
  echo "ERROR: Ollama failed to start after 30 attempts."
  exit 1
fi

# Now that Ollama is running, start the Python script
echo "Ollama is running. Starting main.py..."
python main.py