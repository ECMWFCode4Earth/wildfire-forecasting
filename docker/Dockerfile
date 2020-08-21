## Start from base ubuntu:18.04 or nvidia/cuda:10.2-runtime-ubuntu18.04
FROM nvidia/cuda:10.2-runtime-ubuntu18.04

# Make bash default
SHELL [ "/bin/bash", "--login", "-c" ]

# install packages as root
RUN apt-get update --fix-missing && \
    apt-get install -qq -y wget bzip2 curl git gcc python3-dev build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user, recommended security practice
ARG username=wildfire
ARG uid=1000
ARG gid=100
ENV USER $username
ENV UID $uid
ENV GID $gid
ENV HOME /home/$USER
RUN adduser --disabled-password \
    --gecos "Non-root user" \
    --uid $UID \
    --gid $GID \
    --home $HOME \
    $USER

# Copy required files for building conda environment & update permissions
COPY environment.yml /tmp/environment.yml
RUN chown $UID:$GID /tmp/environment.yml 

# entrypoint.sh used for activating conda env when launching jupyter 
COPY entrypoint.sh /usr/local/bin/
RUN chown $UID:$GID /usr/local/bin/entrypoint.sh && \
    chmod u+x /usr/local/bin/entrypoint.sh

# install latest miniconda as non-root user in $HOME/miniconda3
USER $UID
ENV CONDA_DIR $HOME/miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh

# make non-activate conda commands available
ENV PATH=$CONDA_DIR/bin:$PATH

# make conda activate command available from /bin/bash --login shells
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile

# init bash with conda
RUN conda init bash

# create a project directory inside user home
ENV PROJECT_DIR $HOME/app
RUN mkdir $PROJECT_DIR
WORKDIR $PROJECT_DIR

# build the conda environment
ENV ENV_PREFIX $PROJECT_DIR/env
RUN conda update --name base --channel defaults conda && \
    conda env create --prefix $ENV_PREFIX --file /tmp/environment.yml --force && \
    conda clean --all --yes

# copy necessary files from repo to $HOME/app/ in container
# COPY src $PROJECT_DIR/src
# COPY data $PROJECT_DIR/data
# COPY docs $PROJECT_DIR/docs
# COPY examples $PROJECT_DIR/examples
COPY README.md $PROJECT_DIR/README.md
COPY Code_Structure_Overview.md $PROJECT_DIR/Code_Structure_Overview.md

#change permissions on files/directories to allow edits during training
# USER root
# RUN chown -R $UID:$GID $PROJECT_DIR/data && chmod -R 755 $PROJECT_DIR/data
# RUN chown -R $UID:$GID $PROJECT_DIR/docs && chmod -R 755 $PROJECT_DIR/docs
# RUN chown -R $UID:$GID $PROJECT_DIR/examples && chmod -R 755 $PROJECT_DIR/examples
# RUN chown -R $UID:$GID $PROJECT_DIR/src && chmod -R 755 $PROJECT_DIR/src
# USER $UID

# use ENTRYPOINT to activate conda environment at runtime
ENTRYPOINT [ "/usr/local/bin/entrypoint.sh" ]

# default command will launch JupyterLab server for development with no password/token.
# --generate-config ensure jupyter config file is created before first launch
CMD ["jupyter" "lab" "--generate-config"]
CMD ["jupyter", "lab", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]  