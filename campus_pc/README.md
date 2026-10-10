<!-- Same content as APNVI_Stage2_Setup_Guide.pdf in the planning repo. DRAFT: not yet tested on the real hardware. -->

# Stage 2 Setup Guide — step by step

Follow the steps in order. Each step tells you what to **do**, gives you a grey box to
**copy and paste**, and tells you what you **should see**. If you do not see it, read
**If not** and stop there. Write down the exact message on the screen and send it to Adel.

**How to copy a grey box:** select the text inside it with the mouse, press **Ctrl + C**,
click where you want it, press **Ctrl + V**, then press **Enter**. Paste one box at a time.

| Part | Where | About how long |
|---|---|---|
| A | The campus PC (Windows) | 1 hour, mostly waiting for downloads |
| B | The Jetson | 30 minutes |
| C | Optional: make speech faster on the campus PC | 20 minutes |

**What you need before starting:** the campus PC with internet, an administrator password
for it, Adel's GitHub login, the Jetson with its screen and keyboard, Mo'ath's photos, the
microphone.

---

## Part A — the campus PC

### A1. Open the Command Prompt

**Do:** click the Start button (bottom left), type `cmd`, and press Enter.

**You should see:** a black window with text like `C:\Users\...>` and a blinking cursor.
This is the **Command Prompt**. Most grey boxes below get pasted here.

### A2. Write down the PC's address

**Do:** paste this into the Command Prompt.

```
ipconfig
```

**You should see:** several lines. Find the line **IPv4 Address** and write down the number
after it, for example `10.20.30.40`. This guide calls it **PC_IP**. You need it in Part B.

**If not:** if there are several IPv4 lines, use the one under the section that says
Ethernet or Wi-Fi and is not "Media disconnected".

### A3. Install Ollama (the program that runs the AI)

**Do:** open a web browser, go to **ollama.com/download**, click **Download for Windows**,
open the downloaded file and click **Install**. Accept every question.

**You should see:** a small llama icon near the clock (bottom right). You may need to click
the little arrow ^ to see it.

### A4. Download the AI model

**Do:** go back to the Command Prompt and paste:

```
ollama pull qwen3-vl:4b
```

**You should see:** a progress bar. It downloads about 3 GB. At the end it says
**success**.

**If not:** if it says `'ollama' is not recognized`, close the Command Prompt, open a new
one (step A1), and try again.

### A5. Let the Jetson talk to Ollama

Ollama only answers this PC unless we tell it otherwise.

**Do:**
1. Right-click the llama icon near the clock and click **Quit Ollama**.
2. Click Start, type `environment variables`, and click **Edit environment variables for
   your account**.
3. In the top half, click **New...**
4. Variable name: `OLLAMA_HOST`   Variable value: `0.0.0.0`   Click **OK**, then **OK**.
5. Click Start, type `Ollama`, and open it again.

**You should see:** the llama icon is back near the clock.

### A6. Check Ollama works

**Do:** in the web browser, go to this address:

```
http://localhost:11434
```

**You should see:** a page that only says **Ollama is running**.

**If not:** wait 30 seconds and refresh. If it still fails, repeat A5.

### A7. Install Python

**Do:** in the browser go to **python.org/downloads/windows**, click the newest
**Python 3.11** release, and download **Windows installer (64-bit)**. Open it.
**Important: on the first screen tick the box "Add python.exe to PATH"**, then click
**Install Now**.

**You should see:** "Setup was successful". Click Close.

### A8. Download our code

**Do:**
1. In the browser go to **github.com/adelahmad2005/apnvi-stage1** and log in with
   Adel's GitHub account.
2. Click the green **Code** button, then **Download ZIP**.
3. Make a folder called `apnvi` directly on the C: drive (open **This PC**, **Local Disk
   (C:)**, right-click, **New**, **Folder**).
4. Right-click the downloaded ZIP, click **Extract All...**, type `C:\apnvi`, click
   **Extract**.

**You should see:** a folder `C:\apnvi\apnvi-stage1-main` with folders like `campus_pc`
and `tools` inside.

**Before leaving the PC:** log out of GitHub in the browser.

### A9. Install what the speech program needs

**Do:** close the Command Prompt, open a new one (step A1), and paste these two lines,
one at a time:

```
cd /d C:\apnvi\apnvi-stage1-main\campus_pc
```

```
python -m pip install -r requirements.txt
```

**You should see:** many lines scroll past, ending with **Successfully installed ...**

**If not:** if it says `'python' is not recognized`, Python was installed without the
"Add python.exe to PATH" tick. Run the Python installer again, choose **Modify**, and make
sure that option is on.

### A10. Start the speech program

**Do:** in the same Command Prompt, paste:

```
python speech_server.py --device cpu
```

**You should see:** "Loading Whisper..." and then, after it downloads its model the first
time, **Ready. Listening on port 8090**.

**Leave this window open.** Closing it stops the speech program. Open a new Command Prompt
for the next steps.

*(This uses the processor. Part C makes it faster with the graphics card, later.)*

### A11. Check the speech program works

**Do:** in the browser go to:

```
http://localhost:8090/health
```

**You should see:** a line containing `"ok": true`.

### A12. Open the firewall for the Jetson

**Do:** click Start, type `cmd`, **right-click** Command Prompt and choose **Run as
administrator**. Enter the administrator password. Then paste these two lines, one at a
time:

```
netsh advfirewall firewall add rule name="APNVI Ollama" dir=in action=allow protocol=TCP localport=11434
```

```
netsh advfirewall firewall add rule name="APNVI speech" dir=in action=allow protocol=TCP localport=8090
```

**You should see:** **Ok.** after each line.

**If not:** "The requested operation requires elevation" means this window is not running
as administrator. Close it and open it again with right-click, Run as administrator.

### A13. Try the AI on the test photos

**Do:**
1. Copy Mo'ath's photos into a new folder `C:\apnvi\photos`.
2. Open a new Command Prompt (step A1) and paste, one at a time:

```
python -m pip install numpy opencv-python
```

```
cd /d C:\apnvi\apnvi-stage1-main
```

```
python tools\try_finder.py C:\apnvi\photos
```

**You should see:** one line per photo, with `found=yes` or `found=no` and the seconds it
took. The first photo is slow because the AI is loading.

**Then:** open the folder `C:\apnvi\photos\results`. Each photo has a red box drawn where
the AI thinks the thing is. Count how many boxes are on the right thing and write it down.

**If answers take more than about 3 seconds each:** try the smaller AI. Paste these, one
at a time, and remember to tell Adel you used it:

```
ollama pull qwen3-vl:2b
```

```
python tools\try_finder.py C:\apnvi\photos --model qwen3-vl:2b
```

**Part A is done.** Keep the speech window from A10 open.

---

## Part B — the Jetson

### B1. Open a Terminal

**Do:** on the Jetson press **Ctrl + Alt + T**.

**You should see:** a window with text ending in `$`. This is the **Terminal**. Paste here
with **Ctrl + Shift + V** (not Ctrl + V).

Make sure the Jetson is on the campus wifi (wifi icon, top right).

### B2. Check the Jetson can reach the PC

**Do:** paste these two lines, one at a time. **Replace PC_IP with the number from A2**
before pressing Enter.

```
curl http://PC_IP:11434
```

```
curl http://PC_IP:8090/health
```

**You should see:** the first prints **Ollama is running**. The second prints a line with
`"ok": true`.

**If not:** if it waits for a long time or says "refused": check A5 and A12 on the PC, and
that the speech window from A10 is still open.

### B3. Get the new code and build it

**Do:** paste these lines, one at a time. Enter the Jetson's password if asked (nothing
shows while you type it, that is normal).

```
cd ~/apnvi_demo/src/apnvi-stage1 && git pull
```

```
sudo apt install -y alsa-utils espeak-ng
```

```
source /opt/ros/humble/setup.bash && cd ~/apnvi_demo && colcon build && source install/setup.bash
```

**You should see:** at the end of the last one, **Summary: 2 packages finished**.

**If not:** copy the red error lines and send them to Adel.

### B4. Put the new program on the ESP32 (Mo'ath)

**Do:**
1. On the laptop with the Arduino app, download the code: go to
   **github.com/adelahmad2005/apnvi-stage1**, log in, click **Code**, then **Download ZIP**,
   and extract it. Open the file `esp32\apnvi_esp32\apnvi_esp32.ino` in the Arduino app.
2. Click **Verify** (the tick). Wait for **Done compiling**.
3. Plug in the ESP32, click **Upload** (the arrow). Wait for **Done uploading**.
4. Open **Tools**, **Serial Monitor**, and set the speed to **115200**.

**You should see:** numbers like `1.23` scrolling, as before. Press the button: a line
**B1** appears. Let go: **B0** appears.

5. Close the Serial Monitor and plug the ESP32 back into the Jetson.

### B5. Find the microphone's name

**Do:** plug the microphone into the Jetson and paste:

```
arecord -l
```

**You should see:** a list of cards. Find the microphone's line, for example
`card 2: Device [USB Audio Device], device 0`. Its name is **plughw:** then the card
number, a comma, and the device number. In this example: `plughw:2,0`. This guide calls it
**MIC**.

### B6. Test the microphone

**Do:** paste this (replace MIC), and speak for 3 seconds:

```
arecord -D MIC -f S16_LE -r 16000 -c 1 -d 3 test.wav && aplay test.wav
```

**You should see:** after 3 seconds, you hear your own voice played back.

### B7. Start everything

**Do:** paste this, replacing **PC_IP** and **MIC**:

```
ros2 launch apnvi_stage2 stage2.launch.py pc:=PC_IP mic:=MIC
```

**You should see:** many lines scroll, including **Ready. Qwen at http://...**
**Leave this window open.**

(If you used the smaller AI in A13, add ` model:=qwen3-vl:2b` at the end.)

### B8. Open the screens

**Do:** on a laptop on the same wifi, open the browser. You need the Jetson's address:
open a new Terminal on the Jetson (Ctrl + Alt + T) and paste `hostname -I`. Use the first
number. Then go to:

```
http://JETSON_IP:8081
```

**You should see:** the live colour camera and a green **READY** box underneath.
(Stage 1's screen is still at port 8080.)

### B9. Ask by typing (first test)

**Do:** in the second Jetson Terminal, paste these, one at a time:

```
source ~/apnvi_demo/install/setup.bash
```

```
ros2 topic pub --once /voice/question std_msgs/String "{data: 'where is the exit'}"
```

**You should see:** on the screen: THINKING, then the picture with a red box around the
exit, or the device says **"I can't see it, turn slowly and try again"**.

### B10. Ask with the button (real test)

**Do:** point the camera at something on the list, **hold** the button, say "where is the
exit", **let go**.

**You should see:** LISTENING while you hold, THINKING after you let go, then the red box
(silent) or the "turn slowly" sentence.

### B11. Safety check

**Do:** turn the Jetson's wifi off. Walk the device toward a chair.

**You should see:** Stage 1 still beeps faster as it gets closer. Ask a question with the
button: the device says **"I can't reach the helper computer"**. Turn the wifi back on.

**All done.** Write down for each target: did the box land on it, and how many seconds it
took (shown on the screen).

---

## Part C — optional: faster speech using the graphics card

Only do this after everything above works.

**Do:**
1. Go to **github.com/SYSTRAN/faster-whisper** and find the section **"Download the
   libraries from Purfview's repository"**. Follow its link and download the archive it
   names for Windows.
2. Extract it into a new folder `C:\apnvi\cuda_libs`.
3. Click Start, type `environment variables`, open **Edit environment variables for your
   account**, click **Path** in the top half, **Edit...**, **New**, type
   `C:\apnvi\cuda_libs`, **OK**, **OK**.
4. Close the speech window from A10. Open a new Command Prompt and paste, one at a time:

```
cd /d C:\apnvi\apnvi-stage1-main\campus_pc
```

```
python speech_server.py
```

**You should see:** **Ready. Listening on port 8090**, with "on cuda" in the line above it.

**If not:** any error mentioning cuBLAS, cuDNN or CUDA means it did not work. Go back to
`python speech_server.py --device cpu`, which works fine, only slower.
