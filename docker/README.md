# Using Docker:
Make sure you have the latest docker engine (19.03) installed and docker set up to run in [rootless mode](https://docs.docker.com/engine/security/rootless/). The source code files from this repository are not copied into the docker image but are mounted on the container at the time of launch. The instructions below outline this process, first for CPU based usage and then for GPUs. 

## CPU
To spin up a container for operation on CPUs only, we use [`docker-compose`](https://docs.docker.com/compose/install/). Update the file `.env` inside the `docker/` directory containing the current user details. On ubuntu you can find this `uid` by running `id`. On a system with the username `esowc` and uid `1001`, this `.env` file looks as below:
```
USER=esowc
UID=1001
```
Read more about `docker-compose` environment variables [here](https://docs.docker.com/compose/environment-variables/#the-env-file). Creating a container with the same `UID` will allow us to read and edit the mounted volumes from inside the container. Once you have the `.env` file ready, you only need to run the following command from inside `docker/`:
```bash
docker-compose up --build
```
This command will build your image and launch a container that listens on port `8080`. It also mounts the `data/`, `docs/`, `examples/` and `src/` directories to the container into `/home/$USER/app/`. The container automatically starts  a `jupyterlab` server with authentication disabled. You can now access the containerised environment with all the repository code at `localhost:8080`. If you are SSH-ing into a remote system where you have the docker host and the container, just enable port forwarding during SSH using `ssh -L 8080:localhost:8080 username@hostname` and you should be able to access the JupyterLab server locally at `localhost:8080`.

Note: The final docker image is big (6GB+) and takes time (all those `conda` dependencies) to build. 

## GPU
Docker now has [native support for NVIDIA GPU](https://github.com/NVIDIA/nvidia-docker). To use GPUs with Docker, make sure you have installed the [NVIDIA driver](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#how-do-i-install-the-nvidia-driver) and Docker 19.03 for your Linux distribution on the host machine. It is recommended that you follow the steps outlined in the above **CPU** section to ensure the image is correctly built with all the proper usernames, file system privileges and mounted volumes. Assuming you have the container running from above and you have tested that you can correctly read and write files (run the `src/train.py` script to check), follow the steps below from the **root of the repository**:

```bash
# shut down running container
docker-compose down
# check available docker images
docker image ls
# launch container with gpu and mount volumes
docker run -p 8080:8888 --rm --tty --name esowc-wildfire --volume $(pwd)/data:/home/$USER/app/data --volume $(pwd)/docs:/home/$USER/app/docs --volume $(pwd)/examples:/home/$USER/app/examples --volume $(pwd)/src:/home/$USER/app/src --gpus all --shm-size 10G --detach docker_esowc-wildfire:latest
```
In the last command, replace `$USER` with the value you set to `USER` in the `.env` file in the **CPU** section above. However, there seems to be a lot going on with that last command above. To break it down, we publish a container that:
* listens on host port 8080 - `[-p]`
* is automatically removed when shut down - `[--rm]`
* allocates a pseudo-TTY - `[--tty]`
* is named esowc-wildfire - `[--name]`
* has `data/`, `docs/`, `examples/` and `src/` from this repo in the host filesystem mounted on `/home/$USER/app/` - `[--volume]`
* uses available GPUs - `[--gpus]`
* Increases the Docker shared memory to 10GB to address [this](https://github.com/pytorch/pytorch/issues/2244#issuecomment-318864552) - `[--shm-size]`
* detaches from current session and runs in background - `[--detach]`
* uses the previously built docker image - `[docker_esowc-wildfire:latest]`

You can now access the JupyterLab on `localhost:8080` with all the goodness of GPUs for training, testing and inference.

**Note**: If you run into issues with shared memory while performing inference or training, use the `--shm-size` flag in `docker run` or the run time `shm_size` argument in `docker-compose` to increase the shared memory allocated to Docker. You can also set `--ipc=host` to alleviate any shared memory issues in PyTorch worker pids.

