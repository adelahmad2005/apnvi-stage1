#!/usr/bin/env bash
# APNVI Stage 1 - start the demo. After jetson_setup.sh, just type:  apnvi
# Stop it with Ctrl+C.

WS="$HOME/apnvi_demo"
line () { echo "--------------------------------------------------------------"; }

source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash" 2>/dev/null || { echo "Setup not done yet. Run the setup line first."; exit 1; }

clear
line
echo " APNVI STAGE 1 DEMO"
line

# Camera plugged in? (Intel RealSense shows up as an Intel USB device)
if ! command -v lsusb >/dev/null || lsusb | grep -qiE "8086:0b|realsense"; then
  echo " [OK]   Camera found."
  CAMERA=true
else
  echo " [!!]   CAMERA NOT FOUND. Plug it into a BLUE USB port, then Ctrl+C and type apnvi again."
  CAMERA=false
fi

# ESP32 plugged in?
PORT=$(ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null | head -1)
if [ -n "$PORT" ]; then
  echo " [OK]   ESP32 found on $PORT."
else
  echo " [--]   ESP32 not plugged in. That is OK: the camera works alone."
  PORT=/dev/ttyUSB0
fi

IP=$(hostname -I | awk '{print $1}')
line
echo " LIVE VIDEO - open this in a web browser:"
echo "   on this computer:            http://localhost:8080"
echo "   on a laptop on the same wifi: http://$IP:8080"
line
echo " Starting... (to stop: press Ctrl+C)"
echo

# Open the live video in this computer's browser after a few seconds.
( sleep 10; xdg-open http://localhost:8080 >/dev/null 2>&1 ) &

ros2 launch apnvi_stage1 stage1.launch.py camera:=$CAMERA port:=$PORT
