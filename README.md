## Installation

### Install packages

```bash
# ROS INSTALLATION
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt install curl # if you haven't already installed curl
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -sudo apt updatesudo apt update
sudo apt update
sudo apt install ros-melodic-desktop-full
echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
source ~/.bashrc

# ROS DEPENDENCIES
sudo apt-get install ros-melodic-rosbridge-suite
sudo apt-get install ros-melodic-slam-gmapping

# ROS PYTHON PACKAGES
pip install --upgrade pip
pip install rospkg pyyaml

# AIMSM DEPENDENCIES
git submodule update --recursive --remote --init
pip install -r fastsam/requirements.txt
pip install -r requirements.txt
```

### Download models

1. Download FastSAM model and place it in `weights` folder
   - [link](https://drive.google.com/file/d/1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv/view?usp=sharing)

## Run main.py

```bash
python main.py
```

## RUN Simulator

### Terminal 1: Instantiate the ROSBridge server

```bash
source /opt/ros/melodic/setup.bash
conda activate melodic
roslaunch rosbridge_server rosbridge_websocket.launch
```

---

### Terminal 2: Start up the RViz

```bash
source /opt/ros/melodic/setup.bash
rviz -d rviz/robot_at_virtualhome.rviz
```

---

### Terminal 3: Run the GMapping node

```bash
source /opt/ros/melodic/setup.bash
rosrun gmapping slam_gmapping scan:=/RobotAtVirtualHome/scan
```

---

### Terminal 4: Run Gradio app

```bash
conda activate aimsm
python app.py
```
