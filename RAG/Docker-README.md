
# <center>🚀 Getting Started with Docker for Your LLM Project: A Beginner-Friendly Guide 🚀</center>

This guide will walk you through setting up and running your LLM project using Docker. Even if you've never used Docker before, this step-by-step instruction will help you get everything up and running smoothly.

## <center>🐳 What is Docker and Why Use It? 🐳</center>

Imagine you want to share your project with someone else, or run it on a different computer.  Sometimes, things that work perfectly on your machine might not work the same way elsewhere. This is often due to differences in software versions, libraries, and system configurations.

**Docker solves this problem!**

Think of Docker as creating a **container** - a lightweight, standalone, and executable package that includes everything your software needs to run: code, runtime, system tools, libraries and settings.  This container is isolated from your host system, ensuring that your application runs the same way, regardless of where it's deployed.

**Why is Docker great for this project?**

*   **Consistency:**  Ensures your LLM project runs the same way for everyone, regardless of their operating system or installed software.
*   **Isolation:** Keeps your project separate from other software on your system, preventing conflicts.
*   **Ease of Use:** Simplifies the setup process, especially for complex projects with dependencies like this LLM environment.
*   **Reproducibility:** Makes it easy to recreate the exact environment needed to run your project in the future.

**Prerequisites:**

Before you begin, make sure you have Docker installed on your system. You can download Docker Desktop from [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop) and follow the installation instructions for your operating system (Windows, macOS, or Linux).

---

## <center>🛠️ Step 1: Preparing the `start.sh` Script 🛠️</center>

The `start.sh` file is a script that we'll use to start your LLM application inside the Docker container.  Sometimes, when scripts are created in Windows and then used in Linux-based environments (like Docker containers), they can have extra characters that cause problems.

To fix this, navigate to the folder containing your `start.sh` file in your Linux terminal (or if you are using Windows, in a terminal like Git Bash that can run Linux commands). Then, run the following command:

```bash
sed -i 's/\r$//' start.sh
```

**Explanation:**

*   `sed`:  This is a powerful stream editor in Linux.
*   `-i`:  This option tells `sed` to edit the file "in-place", meaning it will modify the `start.sh` file directly.
*   `'s/\r$//'`: This is a `sed` command that does a substitution (`s`).
    *   `\r`: Represents a carriage return character, which is sometimes found at the end of lines in Windows text files.
    *   `$`:  Matches the end of a line.
    *   `//`:  Replaces the matched carriage return at the end of the line with nothing (effectively removing it).

**In simple terms, this command cleans up the `start.sh` file to ensure it works correctly in the Docker container's Linux environment.**

---

## <center>🏗️ Step 2: Building the Docker Image 🏗️</center>

Now we're going to build the Docker image. Think of a Docker image as a blueprint or template for creating Docker containers. It contains all the instructions and configurations needed to run your application.

Open your terminal, navigate to the directory where your `Dockerfile` is located, and run this command:

```bash
docker build -t $(id -un)/llm-image:1.0 .
```

**Explanation:**

*   `docker build`: This is the command to build a Docker image.
*   `-t $(id -un)/llm-image:1.0`: This option tags your image with a name.
    *   `-t`: stands for "tag".
    *   `$(id -un)`: This part automatically gets your Linux username. This helps in uniquely naming your image, especially if multiple users are on the same system.
    *   `/llm-image:1.0`: This is the name you are giving to your image (`llm-image`) and a version tag (`1.0`).  Versioning is useful for keeping track of different versions of your image.
*   `.`:  The dot at the end specifies the build context. In most cases, `.` means the current directory. Docker will look for a `Dockerfile` in this directory to build the image.

**What happens during `docker build`?**

Docker reads the instructions in your `Dockerfile` step-by-step. Each line in the `Dockerfile` is executed to create layers in your image. This includes:

*   Downloading a base image (`FROM nvcr.io/nvidia/tensorflow:25.01-tf2-py3`).
*   Installing software (`apt-get install`, `pip install`).
*   Copying files (`COPY`).
*   Setting up configurations (`WORKDIR`, `RUN`, `CMD`).
*   Pulling Ollama models (as defined in your Dockerfile).

**This process might take some time, especially the first time you build the image, as Docker needs to download the base image and all the dependencies.**

---

## <center>🚀 Step 3: Running the Docker Container 🚀</center>

Once the image is built, you can run a Docker container from it. A container is a running instance of an image.

To run your LLM project, use the following command:

```bash
docker stop my_llm_container
docker rm my_llm_container
docker run -it \
	--name my_llm_container \
	--gpus all --shm-size=64g --ulimit stack=67108864 \
    -v /data/$(id -un)/RAG:/app \
    -v /data/$(id -un)/RAG/results/:/results \
	$(id -un)/llm-image:1.0
```

**Let's break down this command:**

*   `docker stop my_llm_container`: This command attempts to stop a container named `my_llm_container` if it's already running. This is a safety step to ensure you're starting fresh. If no container with this name is running, it will simply do nothing and not cause an error.
*   `docker rm my_llm_container`: This command removes a container named `my_llm_container` if it exists.  Like `docker stop`, this is for cleanup and ensures you start with a new container.
*   `docker run`: This is the command to run a new Docker container.
*   `-it`: These are two options combined:
    *   `-i` or `--interactive`: Keeps STDIN (standard input) open even if not attached, allowing you to interact with the container.
    *   `-t` or `--tty`: Allocates a pseudo-TTY, which gives you a terminal-like interface inside the container.
*   `--name my_llm_container`:  This assigns the name `my_llm_container` to your running container, making it easier to manage later.
*   `--gpus all`: This is crucial for GPU-accelerated applications like LLMs. It tells Docker to make all available GPUs accessible inside the container.
*   `--shm-size=64g`: This increases the shared memory size available to the container to 64 gigabytes. Shared memory is sometimes needed for performance, especially in machine learning workloads.
*   `--ulimit stack=67108864`: This increases the stack size limit for processes inside the container.  Stack size is memory used for function calls and local variables, and increasing it can prevent stack overflow errors in some applications.
*   `-v /data/$(id -un)/RAG:/app`: This is a **volume mount**. It creates a link between a directory on your host machine and a directory inside the Docker container.
    *   `-v`: option for volume mount.
    *   `/data/$(id -un)/RAG`: This is the path to a directory on your **host machine**.  It's likely where your project code is located. `$(id -un)` again uses your username to construct a path.
    *   `:/app`: This is the path to a directory **inside the Docker container**. In your `Dockerfile`, you set `WORKDIR /app`, so this mounts your project code into the `/app` directory inside the container.  **Any changes you make to files in `/data/$(id -un)/RAG` on your host will be reflected in `/app` inside the container, and vice versa.**
*   `-v /data/$(id -un)/RAG/results/:/results`:  Another volume mount, this time for the `results` directory. This ensures that any results generated by your application inside the container (in the `/results` directory) are saved to `/data/$(id -un)/RAG/results/` on your host machine, so you can access them even after the container stops.
*   `$(id -un)/llm-image:1.0`: This specifies which Docker image to use to run the container. It's the image you built in the previous step.

**After running this command, your Docker container will start. The `start.sh` script inside the container will be executed, which in turn will start the Ollama server and your `main.py` script.** You should see output from your application in the terminal.

---

## <center>🔄 Step 4: Re-attaching to a Running Container 🔄</center>

If you close the terminal where your container is running, or if you want to check on its progress later, you can re-attach to it.

To re-attach to a running container named `my_llm_container`, use this command:

```bash
docker attach my_llm_container
```

This will reconnect your terminal to the standard input, output, and error streams of the running container, allowing you to see the ongoing output and interact with it if needed.

**To detach from the container without stopping it, press `Ctrl+p` then `Ctrl+q`.** The container will continue running in the background.

---

## <center> 🐚 Step 5: Starting the Container in Interactive Mode (Bash) 🐚</center>

Sometimes you might want to enter the Docker container's shell to explore the environment, debug issues, or run commands directly inside the container without automatically starting your `main.py` script.

You can do this by running the container in interactive mode and overriding the default command to start a bash shell:

```bash
docker stop my_llm_container
docker rm my_llm_container
docker run -it --entrypoint /bin/bash $(id -un)/llm-image:1.0
```

**Explanation:**

*   `--entrypoint /bin/bash`: This option overrides the default command specified in the `Dockerfile` (which is to run `start.sh`). Instead, it sets the entry point to `/bin/bash`, which starts a bash shell inside the container.

After running this command, you will be inside the container's bash shell. You can then explore the file system, run commands, and manually start your application components if needed. To exit the container, type `exit` and press Enter.

---

## <center> 🖼️ Step 6: Listing Local Docker Images 🖼️</center>

To see a list of all Docker images that are currently stored on your system, use this command:

```bash
docker image list -a
```

**Explanation:**

*   `docker image list`: This is the command to list Docker images.
*   `-a`: This option stands for "all". It shows all images, including intermediate images and images that are not tagged. Without `-a`, it will only show top-level images.

The output will show you information about each image, including its repository, tag, image ID, creation date, and size.

---

## <center> 🗑️ Step 7: Removing a Docker Image 🗑️</center>

If you want to remove a Docker image from your system to free up disk space, you can use the `docker rmi` command (rmi stands for "remove image").

First, list your images using `docker image list -a` to find the name or ID of the image you want to remove. Then, use the following command, replacing `$(id -un)/my-tf-image:1.0` with the actual image name or ID:

```bash
docker rmi $(id -un)/my-tf-image:1.0
```

**Explanation:**

*   `docker rmi`: This is the command to remove a Docker image.
*   `$(id -un)/my-tf-image:1.0`: This is the name of the image you want to remove. **Make sure you replace this with the correct image name or ID from your `docker image list` output.**

**Important:** Be careful when removing images. Once an image is removed, it's gone unless you have a backup or can rebuild it from the `Dockerfile`. You cannot remove an image if there are containers currently running that are based on that image. You need to stop and remove the containers first.

---

## <center> 🧹 Step 8: Removing All Local Docker Images 🧹</center>

If you want to remove **all** Docker images from your local system, you can use a combination of commands:

```bash
docker images -q | xargs docker rmi
```

**Explanation:**

*   `docker images -q`: This command lists all Docker image IDs (`-q` stands for "quiet", which only outputs the IDs).
*   `|`: This is a pipe symbol. It takes the output of the command on the left (`docker images -q`) and sends it as input to the command on the right (`xargs docker rmi`).
*   `xargs docker rmi`:  `xargs` builds and executes command lines from standard input. In this case, it takes the list of image IDs from `docker images -q` and runs `docker rmi` for each image ID, effectively removing all of them.

**Use this command with caution!** It will delete all your local Docker images.

---

## <center> 🧽 Step 9: Cleaning Up Docker System 🧽</center>

Over time, Docker can accumulate stopped containers, unused networks, "dangling" images (images that are no longer tagged and are not associated with any container), and build cache. These can take up disk space.

To clean up your Docker system and remove these unused resources, you can use the `docker system prune` command:

```bash
docker system prune
```

**Explanation:**

*   `docker system prune`: This command cleans up unused Docker resources.

When you run this command, Docker will ask you for confirmation before removing anything. It will typically remove:

*   All stopped containers.
*   All networks not used by at least one container.
*   All dangling images.
*   Optionally, all unused images (if you add the `-a` flag, be careful!).
*   Build cache.

This is a good command to run periodically to keep your Docker system clean and reclaim disk space.

---

<center> 🎉 **Congratulations!** 🎉 </center>

You've now learned how to use Docker to set up and run your LLM project! You should be able to build the Docker image, run containers, manage them, and clean up your Docker environment.  This knowledge will be valuable for this project and for any future projects that can benefit from containerization. Happy coding!
```