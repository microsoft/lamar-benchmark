# Create a directory for all the dependencies
mkdir external
cd external

#######################
## Installing Ceres-Solver with tag 2.1.0
## Check webpage https://ceres-solver.googlesource.com/ceres-solver/+/refs/tags/2.1.0 for more details
git clone --no-checkout https://ceres-solver.googlesource.com/ceres-solver ceres_v2.1.0
cd  ceres_v2.1.0
git checkout f68321e7de8929fbcdb95dd42877531e64f72f66

# Installing system-wide dependencies for Ceres
# Commands from http://ceres-solver.org/installation.html
echo "Installing system-wide dependencies cmake  build-essential libgoogle-glog-dev libgflags-dev libatlas-base-dev libeigen3-dev libsuitesparse-dev"
sudo apt-get install cmake libgoogle-glog-dev libgflags-dev libatlas-base-dev libeigen3-dev libsuitesparse-dev  build-essential

mkdir build
cd build
cmake ..
make -j4
sudo make install
cd .. # exit build folder
cd .. # Exiting ceres_v2.1.0


#######################
## Installing COLMAP version 3.8
git clone https://github.com/colmap/colmap colmap_v3.8
cd colmap_v3.8
git checkout 3.8
git submodule update --init --recursive

# # Installation instructions from https://colmap.github.io/install.html#build-from-source
echo "Installing system-wide dependencies 
    ninja-build \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libgmock-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \ "
sudo apt-get install \
    ninja-build \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libgmock-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \

mkdir build
cd build
cmake .. -GNinja  -DCMAKE_CUDA_ARCHITECTURES=all
ninja
sudo ninja install

cd .. # exit build folder
cd .. # exit colmap_v3.8

#######################
## Installing HLOC version 1.4
git clone --recursive https://github.com/cvg/Hierarchical-Localization/ hloc_v1.4
cd hloc_v1.4
git checkout v1.4
git submodule update --init --recursive

# installing
echo "Installing hloc...."
python -m pip install -e .

cd .. # exit hloc_v1.4 folder

### Installing pyceres version 1.0
git clone https://github.com/cvg/pyceres.git pyceres_v1.0
cd pyceres_v1.0
git checkout v1.0
git submodule update --init --recursive

# installing pyceres
python -m pip install .
cd .. # exit pyceres 