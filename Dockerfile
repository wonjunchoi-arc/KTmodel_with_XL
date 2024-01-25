ARG UBUNTU_VERSION=20.04
ARG CUDA_VERSION=11.4.3
ARG CUDA=11.4
ARG CUDNN_VERSION=8

FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu${UBUNTU_VERSION}


# ARG CUDA_VERSION
# ARG CUDA
ARG UID=
ARG USER_NAME=
ARG CONDA_ENV_NAME=new

ENV DEBIAN_FRONTEND=noninteractive

#Install the default package for ubuntu
SHELL ["/bin/bash", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ccache \
    cmake \
    curl \
    git \
    libfreetype6-dev \
    libhdf5-serial-dev \
    libzmq3-dev \
    libjpeg-dev \
    libpng-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    pkg-config \
    software-properties-common \
    ssh \
    sudo \
    unzip \
    wget
RUN rm -rf /var/lib/apt/lists/*

#For CUDA profiling
ENV LD_LIBRARY_PATH /usr/local/cuda-${CUDA}/targets/x86_64-linux/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf && \
    ldconfig

#conda install
ENV LANG C.UTF-8
RUN curl -o /tmp/miniconda.sh -sSL http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -bfp /usr/local && \
    rm /tmp/miniconda.sh
RUN conda update -y conda

# Create a user
RUN adduser $USER_NAME -u $UID --quiet --gecos "" --disabled-password && \
    echo "$USER_NAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USER_NAME && \
    chmod 0440 /etc/sudoers.d/$USER_NAME

# For connecting via ssh
RUN echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config && \
    echo "PermitEmptyPasswords yes" >> /etc/ssh/sshd_config && \
    echo "UsePAM no" >> /etc/ssh/sshd_config

USER $USER_NAME
SHELL ["/bin/bash", "-c"]

# Create the conda environment
COPY environment.yaml /
RUN conda env create -f /environment.yaml
ENV PATH /usr/local/envs/$CONDA_ENV_NAME/bin:$PATH
RUN echo "source activate ${CONDA_ENV_NAME}" >> ~/.bashrc


# Enable jupyter lab
RUN source activate ${CONDA_ENV_NAME} && \
    conda install -c conda-forge jupyterlab
    # jupyter serverextension enable --py jupyterlab --sys-prefix




#docker build -t custom/mydocker:latest --build-arg UID=93 --build-arg USER_NAME=jun -f Dockerfile .
#docker run -it --name test --gpus 0 -p 8888:8888 (로컬의 pwd 경로를 도커의 /workspace 경로와 마운트)-v $PWD:/workspace (도커의 작업폴더를 설정)-w /workspace custom/mydocker:latest /bin/bash 
# docker run -it --name test --gpus 0 -p 8888:8888 -v $PWD:/workspace -w /workspace custom/mydocker:latest /bin/bash 
