FROM nvcr.io/nvidia/tensorflow:25.01-tf2-py3
LABEL $(id -un)

# Install curl (needed for ollama later), upgrade pip, and install matplotlib, git
RUN apt-get update && apt-get install -y curl git && \
    pip install --upgrade pip && \
    pip install matplotlib

RUN apt update # Update package lists again after adding the repository

# Install lspci (or lshw) BEFORE installing ollama
RUN apt install -y lshw

# Install ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# **Explicitly source profile to ensure PATH is updated**
RUN source /root/.profile && source /etc/profile

# Ensure venv is available
RUN apt-get install -y python3-venv

# Set working directory inside the container 
WORKDIR /app

# Explicitly set user to root (though RUN commands are usually root)
USER root
# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy all your script files into the container
#COPY . .
# Use the local script files

# --------------------- Additions for ollama service and model pulling ---------------------

# Create ollama service file
# COPY ollama.service /etc/systemd/system/ollama.service

# Enable ollama service
# RUN systemctl start ollama.service && sleep 10  # Start service and wait for 10 seconds
# --------------------- Additions for ollama model pulling during build ---------------------

# **Start ollama server in the background within a RUN command, wait, then pull models**
RUN ollama serve & sleep 10 && \
    ollama pull gemma3:12b && \
    ollama pull llama3.2:1b && \
    ollama pull llama3.2 && \
    ollama pull llama3.1 && \
    ollama pull phi3:mini-128k && \
    ollama pull phi3:medium-128k && \
    ollama pull gemma2:latest && \
    ollama pull qwen2.5 && \
    ollama pull qwen3:8b && \
    ollama pull deepseek-r1:8b && \
    ollama pull deepseek-r1:1.5b

# --------------------- End of additions ---------------------

# Expose the Ollama API port
# EXPOSE 11434

# Start Ollama server in the background and then run main.py
# Use a shell script to handle the background process and ensure proper startup.
COPY start.sh .
RUN sed -i 's/\r$//' /app/start.sh
RUN chmod +x /app/start.sh
RUN ls -l /app/start.sh  # <--- VERIFY PERMISSIONS
      
CMD ["bash", "/app/start.sh"]

    