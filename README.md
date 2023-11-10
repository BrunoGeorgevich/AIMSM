## Installation

### Install packages

```bash
pip install --upgrade pip

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
rviz -d /home/bruno/Documents/RViz\ Settings/robot_at_virtualhome.rviz
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
