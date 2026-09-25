#!/usr/bin/env bash
# APNVI Stage 1 - one-time setup. Safe to run again: it only adds what is missing.
# Run it with this one line (Ubuntu 22.04, e.g. the Jetson on JetPack 6.2):
#   wget -qO- https://raw.githubusercontent.com/adelahmad2005/apnvi-stage1/main/tools/jetson_setup.sh | bash

REPO_URL="https://github.com/adelahmad2005/apnvi-stage1.git"
WS="$HOME/apnvi_demo"                    # everything goes in this folder
say () { echo; echo "========== $* =========="; }
fail () { echo; echo "XXXXXXXXXX  PROBLEM: $*  XXXXXXXXXX"; echo "Take a photo of this screen and send it to Adel."; exit 1; }

# Everything is inside main() so the whole file is read before anything runs.
main () {
  say "APNVI SETUP STARTED - this can take 10 to 30 minutes. Do not close this window."

  # 1. Right Ubuntu?
  . /etc/os-release
  [ "$VERSION_ID" = "22.04" ] || fail "This needs Ubuntu 22.04 (JetPack 6.2). This computer has Ubuntu $VERSION_ID."

  # 2. Internet?
  wget -q --spider https://github.com || fail "No internet. Connect to wifi and run the line again."

  # 3. Password once, at the start
  echo "Type your password below and press Enter (nothing shows while you type, that is normal):"
  sudo -v || fail "Wrong password."

  # 4. ROS 2 Humble (skipped if already installed)
  if [ ! -f /opt/ros/humble/setup.bash ]; then
    say "Installing ROS 2 Humble (the longest part)"
    sudo apt-get update -y && sudo apt-get install -y software-properties-common curl locales \
      || fail "Could not install basic tools."
    sudo locale-gen en_US en_US.UTF-8 >/dev/null
    sudo add-apt-repository -y universe >/dev/null
    V=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
        | grep -F tag_name | awk -F\" '{print $4}')
    curl -fsL -o /tmp/ros2-apt-source.deb \
      "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${V}/ros2-apt-source_${V}.jammy_all.deb" \
      || fail "Could not download the ROS 2 source file."
    sudo dpkg -i /tmp/ros2-apt-source.deb >/dev/null || fail "Could not add the ROS 2 source."
    sudo apt-get update -y
    sudo apt-get install -y ros-humble-ros-base || fail "Could not install ROS 2 Humble."
  else
    say "ROS 2 Humble is already installed - good"
  fi

  # 5. Everything else the demo needs
  say "Installing the camera driver, sound and tools"
  sudo apt-get install -y ros-dev-tools ros-humble-realsense2-camera ros-humble-launch-ros \
    espeak-ng python3-serial pulseaudio-utils alsa-utils git python3-numpy python3-zstd \
    || fail "Could not install the camera driver or tools."
  python3 -c "import cv2" 2>/dev/null || sudo apt-get install -y python3-opencv \
    || fail "Could not install OpenCV."
  sudo usermod -aG dialout "$USER"         # permission to read the ESP32's USB port

  # 6. Get the latest code
  say "Downloading the APNVI code"
  mkdir -p "$WS/src"
  if [ -d "$WS/src/apnvi-stage1/.git" ]; then
    git -C "$WS/src/apnvi-stage1" pull --ff-only || fail "Could not update the code."
  else
    git clone "$REPO_URL" "$WS/src/apnvi-stage1" || fail "Could not download the code."
  fi

  # 7. Build
  say "Building"
  source /opt/ros/humble/setup.bash
  cd "$WS" && colcon build 2>&1 | tail -3
  [ -d "$WS/install/apnvi_stage1" ] || fail "The build did not work."

  # 8. Short command to start the demo: apnvi
  LINE="alias apnvi='bash $WS/src/apnvi-stage1/tools/start_demo.sh'"
  grep -qxF "$LINE" ~/.bashrc || echo "$LINE" >> ~/.bashrc

  say "SETUP FINISHED - ALL GOOD"
  echo "Now: 1) RESTART the Jetson once (needed for the USB permission)."
  echo "     2) Open a terminal and type:  apnvi   then press Enter."
}

main "$@"
