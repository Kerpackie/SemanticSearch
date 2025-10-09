## How to Build and Run 

1.  **Build the Docker Image:**
    Open your terminal in the project root and run:

    ```bash
    docker build -t embedding-server .
    ```

2.  **Run the Docker Container:**
    This is the most important step. We will map the port and, crucially, **use a Docker Volume to persist the Hugging Face model cache**. This means the model is only downloaded once and will be reused across container restarts.

    ```bash
    docker run -p 50051:50051 \
      -v hf_cache:/huggingface/cache \
      --name my-embedding-server \
      -d \
      embedding-server
    ```

    **Command Breakdown:**

    * `-p 50051:50051`: Maps your local port 50051 to the container's port 50051.
    * `-v hf_cache:/huggingface/cache`: Creates (or reuses) a named volume called `hf_cache` and mounts it to the `/huggingface/cache` directory inside the container. This is where the model files will be stored persistently.
    * `--name my-embedding-server`: Gives your container a memorable name.
    * `-d`: Runs the container in detached (background) mode.
    * `embedding-server`: The name of the image you built.

You can check the server logs with `docker logs my-embedding-server`. The first time you run it, you'll see the model being downloaded. Subsequent runs will be much faster as the model will be read from the `hf_cache` volume.