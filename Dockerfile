FROM ubuntu:20.04

# Update package list, upgrade system, set default locale and timezone
RUN apt-get update && apt-get upgrade -y
RUN apt-get install locales
RUN locale-gen en_US en_US.UTF-8
ENV LC_ALL=en_US.UTF-8 
ENV LANG=en_US.UTF-8
ENV TZ=Africa/Lagos

# Disable terminal interactivity
ENV DEBIAN_FRONTEND=noninteractive

# Install python3 and related packages
RUN apt-get update
RUN apt-get install python3-dev python3 python3-pip -y
RUN pip3 install --upgrade pip

# Install build tools and dependencies required by opencv
RUN apt-get install build-essential cmake git pkg-config libgtk-3-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
    gfortran openexr libatlas-base-dev python3-numpy libtbb2 \
    libtbb-dev libdc1394-22-dev libopenexr-dev \
    libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev -y

# Install OpenCV
RUN apt-get install libopencv-dev python3-opencv -y

# Install additional system packages
RUN apt-get install x11-apps -y

# Install python packages for deep learning and more
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

ARG USERNAME=sudoku
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create a non-root user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Create working directory
RUN cd /home/sudoku/
RUN mkdir /sudoku-ai-solver
WORKDIR /sudoku-ai-solver

# Copy chosen files from the repository into the working directory
COPY . /sudoku-ai-solver

# Remove requirements.txt file
RUN rm requirements.txt

ENTRYPOINT [ "jupyter", "lab", "--ip=0.0.0.0", "--port=8890", "--allow-root" , "--no-browser"]

EXPOSE 8890