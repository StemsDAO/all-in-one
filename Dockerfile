# use base image from nvidia/cuda
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu20.04
#FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04
# turn off interactions with base ubuntu image
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app


# Install prerequisites
RUN apt-get update
RUN apt-get install -y wget
RUN apt-get install -y ca-certificates
RUN apt-get install -y libjpeg-dev
RUN apt-get install -y libpng-dev
RUN apt-get install -y curl
RUN apt-get install -y ffmpeg
RUN apt-get install -y git
RUN apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get install -y python3.12 python3.12-venv python3.12-dev python3-pip
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Install the latest version of pip to avoid version conflicts
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12
RUN rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as the default Python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1
RUN update-alternatives --set python /usr/bin/python3.12
RUN python -m ensurepip --upgrade

# Remove any conflicting system packages for requests and urllib3
RUN apt-get remove -y python3-requests python3-urllib3 && apt-get autoremove -y

# Upgrade pip and install any essential Python packages
RUN pip install --upgrade pip setuptools wheel

# Upgrade pip
RUN pip install --upgrade pip

# Verify Python version
RUN python --version
RUN pip --version

COPY requirements.gcp.txt requirements.gcp.txt

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch 2.2 with CUDA 12.1 support
#RUN pip install torch==2.2.0+cu121 -f https://download.pytorch.org/whl/cu121/torch_stable.html
#RUN pip install torchvision==0.17.0+cu121 -f https://download.pytorch.org/whl/cu121/torch_stable.html
#RUN pip install torchaudio==2.2.0+cu121 -f https://download.pytorch.org/whl/cu121/torch_stable.html

# Uninstall simplejson if it is already installed - this is very specific issue with the current set of dependencies
#RUN pip uninstall -y simplejson || true
# Install simplejson before installing other requirements
#RUN pip install --ignore-installed simplejson

# based on https://www.shi-labs.com/natten/ - needed for allin1
RUN pip install natten==0.17.3+torch250cu121 -f https://shi-labs.com/natten/wheels/
RUN pip install git+https://github.com/CPJKU/madmom  # install the latest madmom directly from GitHub

# install rest of python requirements for the fastapi server
RUN pip install -r requirements.gcp.txt

# copy all code files
COPY . .

# start up server
CMD ["uvicorn", "web_api:app", "--host", "0.0.0.0", "--port", "80", "--log-level", "debug"]
