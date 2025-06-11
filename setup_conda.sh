#!/bin/bash

# Check the operating system
OS_NAME=$(uname -s)
OS_VERSION=""

if [[ "$OS_NAME" == "Linux" ]]; then
    if command -v lsb_release &>/dev/null; then
        OS_VERSION=$(lsb_release -rs)
    elif [[ -f /etc/os-release ]]; then
        . /etc/os-release
        OS_VERSION=$VERSION_ID
    fi
    if [[ "$OS_VERSION" != "22.04" ]]; then
        echo "Warning: This script has only been tested on Ubuntu 22.04"
        echo "Your system is running Ubuntu $OS_VERSION."
        read -p "Do you want to continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Installation cancelled."
            exit 1
        fi
    fi
else
    echo "Unsupported operating system: $OS_NAME"
    exit 1
fi

echo "Operating system check passed: $OS_NAME $OS_VERSION"

# Check if the --conda parameter is passed
if [[ "$1" == "--conda" ]]; then
    # Check if an environment name is provided
    if [[ -n "$2" ]]; then
        ENV_NAME="$2"
    else
        ENV_NAME="xr-robotics"
    fi

    # Detect the system's default Python version
    if command -v python3 &>/dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1)
    elif command -v python &>/dev/null; then
        PYTHON_VERSION=$(python --version 2>&1)
    else
        echo "Python is not installed on this system."
        exit 1
    fi

    echo "The system's default Python version is: $PYTHON_VERSION"

    # Extract the major and minor version numbers from the Python version string
    PYTHON_MAJOR_MINOR=$(echo $PYTHON_VERSION | grep -oP '\d+\.\d+')

    # Create a conda environment with the detected Python version
    # Initialize conda
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/anaconda3/etc/profile.d/conda.sh"
    else
        echo "Conda initialization script not found. Please install Miniconda or Anaconda."
        exit 1
    fi

    conda deactivate
    conda remove -n "$ENV_NAME" --all -y
    conda create -n "$ENV_NAME" python=$PYTHON_MAJOR_MINOR -y

    echo "Conda environment '$ENV_NAME' created with Python $PYTHON_MAJOR_MINOR"

    # Activate the conda environment
    conda activate "$ENV_NAME"

    conda deactivate

    echo -e "[INFO] Created conda environment named '$ENV_NAME'.\n"
    echo -e "\t\t1. To activate the environment, run:                conda activate $ENV_NAME"
    echo -e "\t\t2. To install the package, run:                     bash setup_conda.sh --install"
    echo -e "\t\t3. To deactivate the environment, run:              conda deactivate"
    echo -e "\n"

# Check if the --install parameter is passed
elif [[ "$1" == "--install" ]]; then
    # Get the currently activated conda environment name
    if [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
        echo "Error: No conda environment is currently activated."
        echo "Please activate a conda environment first with: conda activate <env_name>"
        exit 1
    fi
    ENV_NAME=${CONDA_DEFAULT_ENV}

    # replace conda c++ dependency with libstdcxx-ng
    if [[ "$OS_NAME" == "Linux" ]]; then
        conda install -c conda-forge libstdcxx-ng -y
    fi
    pip install --upgrade pip

    if [[ "$OS_NAME" == "Linux" ]]; then
        # install compilation tools on conda
        conda install -c conda-forge gcc gxx cmake make -y
    else
        echo "Unsupported operating system: $OS_NAME"
        exit 1
    fi

    # Install the required packages
    rm -rf dependencies
    mkdir dependencies
    cd dependencies

    git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind.git
    cd XRoboToolkit-PC-Service-Pybind
    bash setup_ubuntu.sh

    cd ..

    git clone https://github.com/zhigenzhao/placo.git
    cd placo
    git checkout xml_experimental
    bash setup.sh --install

    cd ../..

    pip install -e . || { echo "Failed to install xrobotoolkit_teleop with pip"; exit 1; }

    echo -e "\n"
    echo -e "[INFO] xrobotoolkit_teleop is installed in conda environment '$ENV_NAME'.\n"
    echo -e "\n"
else
    echo "Invalid argument. Use --conda to create a conda environment or --install to install the package."
    exit 1
fi