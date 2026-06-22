# AI-Focus-Tracker

# 🎯 AI Focus Tracker

A real-time webcam app that watches your eyes and head position, and tells you — instantly — whether you're **Focused**, **Distracted**, or **Drowsy**. No timers, no database, just a live, explainable attention signal powered by computer vision.

Built with **Python**, **Streamlit**, **OpenCV**, and **MediaPipe**.

---

## 📸 What It Looks Like

| Focused | Distracted |
|---|---|
| ![Focused](screenshots/focused.png) | ![Distracted](screenshots/distracted.png) |

| Focused (Blinking) | Drowsy / Eyes Closed |
|---|---|
| ![Blinking](screenshots/blinking.png) | ![Drowsy](screenshots/drowsy.png) |

*(These are illustrative mockups of the app's HUD — your actual video will show your own webcam feed with the same overlay style.)*

---

## ✨ Features

- 📷 Runs entirely in your browser tab — no install of any video software, just your webcam
- 👁️ Detects whether your eyes are open, blinking, or closed using **EAR (Eye Aspect Ratio)**
- 🧭 Detects head turns and tilts using **3D head pose estimation**
- 🚦 Combines both signals into one live status: Focused / Focused (Blinking) / Distracted / Drowsy
- 🎛️ Every threshold is adjustable live from the sidebar — no restart needed
- ⚡ Runs at near real-time speed on a normal laptop CPU (no GPU required)

---

## 🧠 How It Works (Quick Version)

1. Your webcam frame is sent into Python via **WebRTC** (`streamlit-webrtc`).
2. **MediaPipe Face Mesh** finds 468 points on your face.
3. Six points around each eye are used to calculate the **Eye Aspect Ratio (EAR)** — a simple number that drops sharply when an eye closes.
4. Six other points (eye corners, nose tip, mouth corners, chin) are fed into `cv2.solvePnP` to estimate your head's real **3D rotation** (pitch / yaw / roll).
5. Both signals are smoothed over a few frames, then checked in priority order: **Drowsy → Distracted → Blinking → Focused**.
6. The result is drawn directly on the video as a HUD, and mirrored into the sidebar.

> Want the full deep-dive with code explanations? See [`AI-Focus-Tracker-Code-Explained.md`](AI-Focus-Tracker-Code-Explained.md) if it's included in this repo.

---

## 🛠️ Tech Stack

| Tool | Role |
|---|---|
| [Streamlit](https://streamlit.io) | Web UI |
| [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc) | Browser camera ↔ Python video bridge |
| [OpenCV](https://opencv.org) | Image processing, drawing, head pose math |
| [MediaPipe](https://developers.google.com/mediapipe) | Face landmark detection |
| [NumPy](https://numpy.org) | Vector math |
| [PyAV](https://pyav.org) | Video frame decoding |

---

## 📁 Project Structure

```
AI-Focus-Tracker/
├── app.py              # Main application — all the logic lives here
├── requirements.txt    # Python packages needed to run the app
├── README.md           # You are here
├── .gitignore          # Keeps venv/, __pycache__, etc. out of git
└── screenshots/         # Demo images used in this README
```

---

## ✅ Prerequisites

- **Python 3.9–3.11** installed ([download here](https://www.python.org/downloads/))
- A **webcam**
- A **Chrome or Edge browser** (best WebRTC support)

---

## 🚀 Setup — Step by Step

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/AI-Focus-Tracker.git
cd AI-Focus-Tracker
```

### 2. Create a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You'll know it worked because your terminal prompt now starts with `(venv)`.

### 3. Install the required packages

```bash
pip install -r requirements.txt
```

This downloads Streamlit, OpenCV, MediaPipe, and everything else the app needs. It may take a few minutes the first time.

### 4. Run the app

```bash
streamlit run app.py
```

A browser tab should open automatically at `http://localhost:8501`. If it doesn't, copy the URL printed in your terminal into your browser manually.

### 5. Allow camera access

Your browser will ask for camera permission — click **Allow**. You should see your own video feed appear with the status HUD on top within a few seconds.

---

## 🎮 Using the App

- **Sidebar sliders** let you tune sensitivity live:
  - `Yaw threshold` — how sharp a left/right turn counts as "Distracted"
  - `Pitch UP / DOWN threshold` — how far up/down counts as "Distracted"
  - `Drowsy frame count` — how many consecutive closed-eye frames count as "Drowsy"
- **Live Stats** in the sidebar shows your current EAR, pitch/yaw, and blink count.
- The **color legend** at the bottom of the page explains what each HUD color means.
- Click **STOP** to release the camera when you're done.

---

## 🐛 Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| Video is laggy / shows your movement a few seconds late | Detection can't keep up with every frame, frames queue up | Already handled in this version — it only runs full detection every 2nd frame |
| Video stuck on a spinner | Old camera connection didn't close, or another app/tab is using the webcam | Close other camera tabs, hard-refresh (`Ctrl+Shift+R`), try again |
| "Looking right" / "looking down" never triggers | Mirrored webcam flips left/right, or thresholds are strict | This version already corrects for mirroring; lower the relevant slider if it's still too strict |
| `KeyError: st.session_state has no key...` after clearing cache | The webcam component was still active when cache was cleared | Stop the stream first, then clear cache, then hard-refresh the page |
| Page is too tall, need to zoom out to see everything | Streamlit's default top padding + large video size | Already handled via custom CSS and a smaller `video_html_attrs` size in this version |

---

## ⚙️ Customizing Thresholds in Code

All thresholds also exist as class constants in `app.py` (inside `FocusProcessor`) if you want to change the *defaults* rather than adjusting sliders every time:

```python
class FocusProcessor(VideoProcessorBase):
    OPEN_EAR_TH   = 0.25   # EAR ≥ this = eyes open
    BLINK_EAR_TH  = 0.18   # EAR < this = eyes closed
    DROWSY_FRAMES = 8      # consecutive closed-eye frames → "Drowsy"
    YAW_TH        = 20     # left/right turn angle → "Distracted"
    PITCH_UP_TH   = 20     # upward tilt angle → "Distracted"
    PITCH_DOWN_TH = -20    # downward tilt angle → "Distracted"
```

---

## 🗺️ Roadmap (Not Yet Built)

These ideas are **not implemented yet** — they're possible next steps:

- ⏱️ Smart Pomodoro timer that pauses when you're not Focused
- 🙂 Emotion detection (tired / confused / frustrated) via DeepFace
- 💾 SQLite session history
- 📊 Pandas + Plotly analytics dashboard

---

## 📄 License

This project is licensed under the **MIT License** — see [`LICENSE`](LICENSE) for details.

---

## 🙏 Acknowledgments

Built with [MediaPipe](https://developers.google.com/mediapipe) by Google, [OpenCV](https://opencv.org), and [Streamlit](https://streamlit.io).
