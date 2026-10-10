# Monday checklist: getting Stage 2 running

Two parts. Part A sets up the campus PC (Windows 10, RTX 2070), about an hour.
Part B starts Stage 2 on the Jetson. Do them in order. If a step fails, stop and note
the exact error.

Words used below:
- **Command Prompt**: Windows' text window. Start menu, type `cmd`, press Enter.
- **Ollama**: the program that runs the AI model (Qwen) and answers over the network.
- **Speech server**: our `speech_server.py`, which turns a recording into words.

---

## Part A: the campus PC

### A1. Before anything
- [ ] Confirm we are allowed to install programs on this PC. Some steps need an
  administrator password.
- [ ] Find the PC's IP address: in Command Prompt type `ipconfig` and write down the
  **IPv4 Address** (looks like `10.x.x.x`). Everything below calls this `PC_IP`.

### A2. Ollama and Qwen
- [ ] Download the Windows installer from ollama.com and install it.
- [ ] In Command Prompt: `ollama pull qwen3-vl:4b` (downloads about 3 GB).
- [ ] Check it: `ollama list` shows `qwen3-vl:4b`.
- [ ] Let the Jetson reach it. Ollama only answers this PC unless told otherwise:
  1. Quit Ollama: right-click its icon near the clock, then Quit.
  2. Start menu, type `environment variables`, open **Edit environment variables for your
     account**.
  3. **New...**: name `OLLAMA_HOST`, value `0.0.0.0`. OK.
  4. Start Ollama again from the Start menu.
- [ ] Check it: open `http://localhost:11434` in a browser. It says **Ollama is running**.

### A3. Python and the speech server
- [ ] Install Python 3.11 from python.org. On the first installer screen tick
  **Add python.exe to PATH**.
- [ ] Get our code: on GitHub, `adelahmad2005/apnvi-stage1`, **Code**, **Download ZIP**
  (needs Adel's GitHub login). Unzip it, for example to `C:\apnvi`.
- [ ] In Command Prompt:
  ```
  cd C:\apnvi\apnvi-stage1-main\campus_pc
  pip install -r requirements.txt
  ```
- [ ] Graphics card libraries for the speech server. faster-whisper needs two NVIDIA
  libraries (cuBLAS and cuDNN, for CUDA 12) that Windows does not have by default.
  The faster-whisper page (github.com/SYSTRAN/faster-whisper, section "Download the
  libraries from Purfview's repository") links an archive with them. Download it, unzip
  it into a folder such as `C:\apnvi\cuda_libs`, and add that folder to **Path** in the
  same environment variables window as A2.
- [ ] Start it: `python speech_server.py`. The first start downloads the Whisper model.
  Wait for **Ready. Listening on port 8090**.
- [ ] If it fails with an error about cuBLAS, cuDNN or CUDA: stop it (Ctrl+C) and use the
  processor instead, which is slower but needs nothing extra:
  `python speech_server.py --device cpu`
- [ ] Check it: open `http://localhost:8090/health` in a browser. It shows `"ok": true`.

### A4. Let the Jetson in through the Windows firewall
- [ ] Start menu, type `cmd`, right-click **Run as administrator**, then:
  ```
  netsh advfirewall firewall add rule name="APNVI Ollama" dir=in action=allow protocol=TCP localport=11434
  netsh advfirewall firewall add rule name="APNVI speech" dir=in action=allow protocol=TCP localport=8090
  ```
  (If Windows pops up "Allow access?" for Ollama or Python, click Allow.)

### A5. Try the finder on the test photos (on the PC itself)
- [ ] In Command Prompt: `pip install numpy opencv-python`
- [ ] Put Mo'ath's photos in a folder, for example `C:\apnvi\photos`, then:
  ```
  cd C:\apnvi\apnvi-stage1-main
  python tools\try_finder.py C:\apnvi\photos
  ```
- [ ] Open `C:\apnvi\photos\results`. Check each red box by eye and note how many are
  right. `results.csv` has the time each answer took.
- [ ] If answers take more than about 3 seconds, try the smaller model:
  `ollama pull qwen3-vl:2b`, then add `--model qwen3-vl:2b` to the command above.
  If the 2B is used, add `model:=qwen3-vl:2b` to the Jetson command in B5 too.

**Part A done when:** both checks in A2 and A3 pass, and the finder has run on the photos.

---

## Part B: the Jetson

### B1. Can the Jetson reach the PC? (This is Mo'ath's network check)
- [ ] Jetson on campus wifi. In a terminal, with the real PC_IP:
  ```
  curl http://PC_IP:11434
  curl http://PC_IP:8090/health
  ```
  The first prints **Ollama is running**. The second prints `"ok": true`.
  If either hangs or says "refused": check A2 (OLLAMA_HOST), A4 (firewall), and that
  both are on the same network.

### B2. Update and build the code
- [ ] In a terminal:
  ```
  cd ~/apnvi_demo/src/apnvi-stage1 && git pull
  sudo apt install -y alsa-utils espeak-ng
  source /opt/ros/humble/setup.bash
  cd ~/apnvi_demo && colcon build && source install/setup.bash
  ```

### B3. The ESP32 (Mo'ath)
- [ ] Open `esp32/apnvi_esp32/apnvi_esp32.ino` in the Arduino app, click **Verify**
  (it must compile), then **Upload**.
- [ ] Serial Monitor at 115200: distances keep coming as before. Press the button: `B1`.
  Let go: `B0`.

### B4. The microphone
- [ ] Plug it in, then `arecord -l`. Find its line, e.g. `card 2: ... device 0`.
  The name to use is then `plughw:2,0` (card number, device number).
- [ ] Quick test: `arecord -D plughw:2,0 -f S16_LE -r 16000 -c 1 -d 3 test.wav`, speak,
  then `aplay test.wav`. You should hear yourself.

### B5. Start everything and test
- [ ] One command, with the real PC_IP and microphone name:
  ```
  ros2 launch apnvi_stage2 stage2.launch.py pc:=PC_IP mic:=plughw:2,0
  ```
- [ ] Open the Stage 2 view from a laptop on the same wifi: `http://JETSON_IP:8081`.
  Stage 1's view is still on `http://JETSON_IP:8080`.
- [ ] Typed test first, in a second terminal (after `source ~/apnvi_demo/install/setup.bash`):
  ```
  ros2 topic pub --once /voice/question std_msgs/String "{data: 'where is the exit'}"
  ```
  The view shows THINKING, then the picture with the red box, or the device says
  "I can't see it, turn slowly and try again".
- [ ] Button test: hold the button, ask "where is the exit", let go.
- [ ] Safety check: turn the Jetson's wifi off and walk at a chair. Stage 1 still
  beeps and warns. Ask a question: the device says "I can't reach the helper computer".

### If the red box is in the right place on the PC photos but tiny signs are missed live
The live colour picture may be smaller than Mo'ath's photos. Tell Adel: raising the
camera resolution is one setting in the Stage 1 launch file.
