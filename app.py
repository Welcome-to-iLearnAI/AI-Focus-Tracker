import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("aiortc").setLevel(logging.ERROR)
logging.getLogger("aioice").setLevel(logging.ERROR)

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# --------------------------------------------------
# Configure the Streamlit web page
# --------------------------------------------------
# page_title : Text shown in the browser tab
# page_icon  : Emoji shown in the browser tab
# layout     : "wide" uses the full browser width
st.set_page_config(page_title="Study Focus Tracker", page_icon="🎯", layout="wide") 

# --------------------------------------------------
# Custom CSS styling for the Streamlit app
# --------------------------------------------------
# .metric-card : Creates a custom card style
# background   : Dark card background color
# border-radius: Rounded corners
# padding      : Space inside the card
# margin-bottom: Space below the card
# border-left  : Blue line on the left side
#
# h1 styling changes the title color
#
# unsafe_allow_html=True allows Streamlit
# to render raw HTML/CSS code
st.markdown("""
<style>
    .metric-card { background:#1e2130; border-radius:12px; padding:16px 20px;
                   margin-bottom:10px; border-left:4px solid #4c8bf5; }
    h1 { color:#e8eaf6; }
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)
# --------------------------------------------------
# Main application title displayed on the page
# --------------------------------------------------
st.title("🎯 AI Focus Tracker")

# --------------------------------------------------
# Load MediaPipe Face Mesh model
# --------------------------------------------------
# Face Mesh detects 468 facial landmarks
# such as eyes, nose, mouth, and eyebrows
mp_face_mesh = mp.solutions.face_mesh

# --------------------------------------------------
# Load MediaPipe drawing utilities
# --------------------------------------------------
# Used to draw facial landmarks and
# connections on the video frame
mp_drawing   = mp.solutions.drawing_utils

# --------------------------------------------------
# Landmark indices for the LEFT eye
# --------------------------------------------------
# These landmark IDs correspond to points
# around the left eye and are used to
# calculate the Eye Aspect Ratio (EAR)
LEFT_EYE  = [33,  160, 158, 133, 153, 144]

# --------------------------------------------------
# Landmark indices for the RIGHT eye
# --------------------------------------------------
# These landmark IDs correspond to points
# around the right eye and are used to
# calculate the Eye Aspect Ratio (EAR)
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Drawing Specification
# Pre-build thin connection drawing spec once (avoid recreating every frame)
"""Creates a drawing style for MediaPipe landmarks.Light gray color.Very thin lines.Do not draw circles on landmarks."""
MESH_SPEC = mp_drawing.DrawingSpec(color=(180, 180, 180), thickness=1, circle_radius=0)

# Eye Aspect Ratio (EAR)
def eye_aspect_ratio(landmarks, eye_points, w, h):
    """
    Calculate Eye Aspect Ratio (EAR).

    Parameters:
        landmarks  : MediaPipe facial landmarks
        eye_points : Landmark IDs for one eye
        w          : Image width
        h          : Image height

    Returns:
        EAR value (higher = eye open, lower = eye closed)
    """
    # Convert MediaPipe normalized coordinates (0-1)
    # into actual pixel coordinates on the image
    pts = [np.array([landmarks[p].x * w, landmarks[p].y * h]) for p in eye_points]
    # Assign the six eye landmarks to variables
    p1, p2, p3, p4, p5, p6 = pts
    """Distance between upper and lower eyelid (first vertical measurement)
    Distance between upper and lower eyelid (second vertical measurement)
    Distance between left and right eye corners (horizontal measurement)
    """
    return (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * np.linalg.norm(p1 - p4))

# Head Pose Function: Determine if the student is:Looking left/right/up/down
def get_head_pose(landmarks, w, h):
    # Create point lists: Stores:2D image points, 3D face points
    face_2d, face_3d = [], []
    # Select important facial landmarks: These correspond roughly to:These points are enough to estimate head orientation.
    """
        | Landmark | Face Part        |
        | -------- | ---------------- |
        | 33       | Left eye corner  |
        | 263      | Right eye corner |
        | 1        | Nose tip         |
        | 61       | Left mouth       |
        | 291      | Right mouth      |
        | 199      | Chin             |
    """
    for idx in [33, 263, 1, 61, 291, 199]:
        lm = landmarks[idx]
        # Converts MediaPipe coordinates into image pixels.
        x, y = int(lm.x * w), int(lm.y * h)
        # Save 2D coordinates
        face_2d.append([x, y])
        # Save 3D coordinates: z gives depth information.
        face_3d.append([x, y, lm.z])

    face_2d = np.array(face_2d, dtype=np.float64) # OpenCV requires NumPy arrays.
    face_3d = np.array(face_3d, dtype=np.float64)

    # --------------------------------------------------
    # Define camera parameters for head pose estimation
    # --------------------------------------------------

    # Approximate the camera focal length using the image width.
    # Since we usually don't know the real webcam focal length,
    # using the image width is a common approximation.
    focal_length = w 

    # Create the camera matrix (intrinsic camera parameters).
    #
    # [ fx   0   cx ]
    # [  0  fy   cy ]
    # [  0   0    1 ]
    #
    # fx, fy = focal lengths
    # cx, cy = image center coordinates
    cam_matrix = np.array([
        [focal_length, 0, w / 2],
        [0, focal_length, h / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    # Assume no lens distortion from the webcam.
    # This distortion matrix contains all zeros.
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    # --------------------------------------------------
    # Estimate head rotation using OpenCV solvePnP
    # --------------------------------------------------
    #
    # Inputs:
    #   face_3d     -> 3D facial landmark coordinates
    #   face_2d     -> 2D image coordinates
    #   cam_matrix  -> camera settings
    #   dist_matrix -> lens distortion settings
    #
    # Output:
    #   success -> True if pose estimation succeeded
    #   rot_vec -> rotation vector describing head orientation
    #
    success, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix,
                                        flags=cv2.SOLVEPNP_ITERATIVE)
    if not success: # # If pose estimation fails, return zero angles
        return 0.0, 0.0, 0.0

    # --------------------------------------------------
    # Convert rotation vector into a rotation matrix
    # --------------------------------------------------
    #
    # OpenCV initially returns rotation as a vector.
    # Rodrigues converts it into a 3x3 rotation matrix.
    #
    rmat, _ = cv2.Rodrigues(rot_vec)
    # --------------------------------------------------
    # Extract Euler angles from the rotation matrix
    # --------------------------------------------------
    #
    # Angles represent:
    #   Pitch -> head up/down
    #   Yaw   -> head left/right
    #   Roll  -> head tilt
    #
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    # --------------------------------------------------
    # Return head pose angles
    # --------------------------------------------------
    #
    # angles[0] -> Pitch
    # angles[1] -> Yaw
    # angles[2] -> Roll
    #
    # These values are used to determine whether the
    # student is looking at the screen or looking away.
    #
    # NOTE: yaw is negated here because most webcams show a
    # mirrored ("selfie-view") image — without this flip, turning
    # your head right would register as "looking left" in code.
    return angles[0] * 360, -angles[1] * 360, angles[2] * 360


class FocusProcessor(VideoProcessorBase):
    # Thresholds (mutable from sidebar)The user can change these values from the Streamlit sidebar while the app is running.No code changes needed.
    OPEN_EAR_TH   = 0.25
    BLINK_EAR_TH  = 0.18
    DROWSY_FRAMES = 8
    YAW_TH        = 20
    PITCH_UP_TH   = 20
    PITCH_DOWN_TH = -20
    """ Example:
    Frame 1 = 18°
    Frame 2 = 23°
    Frame 3 = 20°
    SMOOTH: (18 + 23 + 20)/3"""
    SMOOTH_N      = 3 # Instead of using only the latest frame: Average them, This reduces jitter.

    def __init__(self):
        """
        Constructor for FocusProcessor.

        Runs once when the video processor starts.
        Initializes MediaPipe Face Mesh, tracking variables,
        counters, and shared state values.
        """
        # KEY FIX: refine_landmarks=False — removes iris tracking, ~3x faster

        # --------------------------------------------------
        # Initialize MediaPipe Face Mesh
        # --------------------------------------------------

        # max_num_faces=1
        #     Track only one face for better performance.
        #
        # refine_landmarks=False
        #     Disable iris tracking.
        #     Makes processing much faster (~3x faster).
        #
        # min_detection_confidence=0.5
        #     Face must be detected with at least 50% confidence.
        #
        # min_tracking_confidence=0.5
        #     Continue tracking only if confidence is at least 50%.
        #
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # --------------------------------------------------
        # History Buffers (for smoothing noisy predictions)
        # --------------------------------------------------
        # Stores the last 3 EAR values.
        # Helps smooth blinking measurements.
        self.ear_history   = deque(maxlen=3)

        # Stores the last 3 pitch values.
        # Helps smooth head up/down movement.
        self.pitch_history = deque(maxlen=3)

        # Stores the last 3 yaw values.
        # Helps smooth head left/right movement.
        self.yaw_history   = deque(maxlen=3)

        # --------------------------------------------------
        # Blink Tracking Variables
        # --------------------------------------------------

        # Counts consecutive frames where eyes are closed.
        self.blink_counter = 0
        # Total number of blinks detected during the session.
        self.blink_total   = 0
        # Counts how many video frames have been processed.
        self.frame_count   = 0

       
        # --------------------------------------------------
        # Shared State Variables
        # --------------------------------------------------
        # These values can be displayed in the Streamlit sidebar
        # and updated every frame.

        # Current attention status
        # Examples:
        # "Focused"
        # "Looking Away"
        # "Drowsy"
        # "No Face"
        self.last_status = "No Face"
        # When the app starts: default value
        self.last_ear    = 0.0 # Most recent Eye Aspect Ratio (EAR)
        # Most recent head pitch angle
        # (looking up/down)
        self.last_pitch  = 0.0
        # Most recent head yaw angle
        # (looking left/right)
        self.last_yaw    = 0.0

        # Cache last result to skip reprocessing on identical frames
        # Stores the last processed frame result (overlay).
        """Think:
        Frame 100 → Processed → Save result
        Frame 101 → Same as Frame 100
        Instead of processing again, the app can reuse the previous result."""
        self._last_overlay = None

        # Landmarks from the last frame where we actually ran face
        # detection. Reused on skipped frames (see recv()) so we
        # don't have to detect a face on every single frame.
        self._last_landmarks = None

    # --------------------------------------------------
    # Process Each Webcam Frame
    # --------------------------------------------------
    # This function is automatically called for every
    # frame captured from the webcam.
    #
    # Input:
    #   frame -> current webcam frame
    #
    # Output:
    #   processed video frame
    #
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # --------------------------------------------------
        # Convert AV frame into a NumPy/OpenCV image
        # --------------------------------------------------
        # AV format → OpenCV format
        #
        # bgr24:
        #   B = Blue
        #   G = Green
        #   R = Red
        #
        img = frame.to_ndarray(format="bgr24")
        # --------------------------------------------------
        # Get image dimensions
        # --------------------------------------------------
        #
        # h = image height
        # w = image width
        # _ = number of color channels (3)
        #
        h, w, _ = img.shape

        # KEY FIX: downscale for detection, draw on original
        # --------------------------------------------------
        # Performance Optimization
        # --------------------------------------------------
        # Instead of running MediaPipe on the full image,
        # process a smaller version to improve speed.
        #
        # 0.5 means:
        #   50% of original size
        #
        scale   = 0.5
        # Resize image to half width and half height
        #
        # Example:
        # Original: 640 x 480
        # Smaller : 320 x 240
        #
        small   = cv2.resize(img, (int(w * scale), int(h * scale)))
        # --------------------------------------------------
        # Convert BGR to RGB
        # --------------------------------------------------
        # OpenCV uses BGR.
        # MediaPipe expects RGB.
        #
        rgb     = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        # --------------------------------------------------
        # Run Face Mesh Detection
        # --------------------------------------------------
        # MediaPipe detects facial landmarks
        # such as:
        #
        # - Eyes
        # - Nose
        # - Mouth
        # - Face outline
        #
        # Only run the expensive face-detection model every other
        # frame. On the frames we skip, we reuse the landmarks from
        # the last frame that WAS detected, instead of detecting a
        # face again. This cuts the heavy work in half so processing
        # keeps up with the camera and the video doesn't fall behind.
        self.frame_count += 1
        if self.frame_count % 2 == 0:
            results = self.face_mesh.process(rgb)
            if results.multi_face_landmarks:
                self._last_landmarks = results.multi_face_landmarks[0]
        # On odd frames, self._last_landmarks just keeps its value
        # from the previous detected frame.
        # --------------------------------------------------
        # Create a copy of the original image
        # --------------------------------------------------
        # All drawings (landmarks, text, status)
        # will be drawn on this image.
        #
        # We use the original full-resolution image
        # so the display remains sharp.
        #
        overlay = img.copy()

        # --------------------------------------------------
        # Check if a face was detected
        # --------------------------------------------------
        if self._last_landmarks is not None:
            # Use the most recently detected face landmarks — either
            # fresh from this frame, or reused from the last frame
            # that was actually detected (if this frame was skipped).
            face_lm = self._last_landmarks

            # Draw minimal contours only (faster than full tessellation)
            # --------------------------------------------------
            # Draw Face Mesh Contours
            # --------------------------------------------------
            # Draw only major face outlines:
            # eyes, eyebrows, mouth, face boundary
            #
            # Using FACEMESH_CONTOURS is much faster
            # than drawing all 468 landmark connections.
            #
            mp_drawing.draw_landmarks(
                overlay, # image to draw on
                face_lm, # detected face landmarks
                mp_face_mesh.FACEMESH_CONTOURS, # contour connections only
                landmark_drawing_spec=None, # do not draw landmark dots
                connection_drawing_spec=MESH_SPEC  # line style
            )
            # --------------------------------------------------
            # Get all facial landmarks
            # --------------------------------------------------
            # lm contains 468 face landmark points
            #
            lm = face_lm.landmark
            # --------------------------------------------------
            # Calculate Left Eye EAR
            # --------------------------------------------------
            left_ear  = eye_aspect_ratio(lm, LEFT_EYE,  w, h)
            # --------------------------------------------------
            # Calculate Right Eye EAR
            # --------------------------------------------------
            right_ear = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
            # --------------------------------------------------
            # Average both eyes
            # --------------------------------------------------
            # Gives a more stable measurement.
            #
            avg_ear   = (left_ear + right_ear) / 2.0
            # --------------------------------------------------
            # Estimate Head Pose
            # --------------------------------------------------
            # Returns:
            # pitch = up/down
            # yaw   = left/right
            # roll  = head tilt
            #
            pitch, yaw, roll = get_head_pose(lm, w, h)

            # --------------------------------------------------
            # Store values in history buffers
            # --------------------------------------------------
            # These deques keep the last few values.
            # Helps reduce noise and sudden jumps.
            #
            self.ear_history.append(avg_ear)
            self.pitch_history.append(pitch)
            self.yaw_history.append(yaw)

            # --------------------------------------------------
            # Smooth values using moving average
            # --------------------------------------------------
            # Example:
            #
            # EAR history:
            # [0.28, 0.30, 0.29]
            #
            # Average:
            # 0.29
            #
            # This makes the output more stable.
            #

            # Smoothed EAR
            s_ear   = float(np.mean(self.ear_history))
            s_pitch = float(np.mean(self.pitch_history)) # Smoothed Pitch
            s_yaw   = float(np.mean(self.yaw_history))# Smoothed Yaw

            # Drowsiness
            # --------------------------------------------------
            # Blink and Drowsiness Detection
            # --------------------------------------------------

            # If EAR is below the blink threshold,
            # eyes are considered closed.
            if avg_ear < self.BLINK_EAR_TH:
                # Increase consecutive closed-eye frame count
                self.blink_counter += 1
            else:
                # Eyes opened again

                # If eyes were closed for at least 2 frames,
                # count it as a blink.
                if self.blink_counter >= 2:
                    self.blink_total += 1
                self.blink_counter = 0 # Reset closed-eye frame counter

            # --------------------------------------------------
            # Determine Current State
            # --------------------------------------------------

            # Drowsy if eyes remain closed for many frames
            drowsy          = self.blink_counter >= self.DROWSY_FRAMES
            eyes_open       = s_ear >= self.OPEN_EAR_TH # Eyes considered open if EAR is above threshold
            looking_left    = s_yaw < -self.YAW_TH
            looking_right   = s_yaw >  self.YAW_TH
            looking_up      = s_pitch >  self.PITCH_UP_TH
            looking_down    = s_pitch <  self.PITCH_DOWN_TH
            # Distracted if looking away in any direction
            pose_distracted = looking_left or looking_right or looking_up or looking_down

            # --------------------------------------------------
            # Determine Status Message and Display Color
            # --------------------------------------------------

            # Highest priority:
            # Drowsiness
            if drowsy:
                status = "Drowsy / Eyes Closed"
                color  = (0, 0, 220) # Red color (BGR format)
            elif pose_distracted: # Second priority:# Head turned away
                if looking_left:
                    status = "Distracted - Looking Left"
                elif looking_right:
                    status = "Distracted - Looking Right"
                elif looking_up:
                    status = "Distracted - Looking Up"
                else:
                    status = "Distracted - Looking Away"
                color = (0, 140, 255) # Orange color (BGR)
            elif not eyes_open:
                status = "Focused (Blinking)"
                color  = (0, 220, 120) # Green color (BGR)
            else:
                status = "Focused"
                color  = (0, 255, 80)# Bright green color (BGR)
 
            # --------------------------------------------------
            # Save Latest Values
            # --------------------------------------------------
            # These values are later displayed
            # in the Streamlit sidebar.
            self.last_status = status
            self.last_ear    = s_ear # Latest smoothed EAR value
            self.last_pitch  = s_pitch # Latest smoothed pitch value
            self.last_yaw    = s_yaw

            # HUD — semi-transparent bar
            # --------------------------------------------------
            # Create HUD (Heads-Up Display)
            # --------------------------------------------------
            # HUD = information panel shown on top of the video
            # displaying status, EAR, blinks, and head pose.
            #

            # Create a copy of the current frame
            bar = overlay.copy()
            # Draw a dark rectangle at the top of the screen
            #
            # Parameters:
            # (0,0)     = top-left corner
            # (w,135)   = bottom-right corner
            # (20,20,30)= dark gray color (BGR)
            # -1        = filled rectangle
            #
            cv2.rectangle(bar, (0, 0), (w, 135), (20, 20, 30), -1)
            # Blend rectangle with original image
            # to create a semi-transparent effect.
            #
            # 50% rectangle + 50% original image
            #
            cv2.addWeighted(bar, 0.5, overlay, 0.5, 0, overlay)
            # --------------------------------------------------
            # Display Current Status
            # --------------------------------------------------
            #
            # Examples:
            # Focused
            # Distracted
            # Drowsy
            #
            cv2.putText(overlay, status,
                        (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2, cv2.LINE_AA)
            # --------------------------------------------------
            # Display EAR and Blink Count
            # --------------------------------------------------
            #
            # Example:
            # EAR: 0.29  Blinks: 15
            #
            cv2.putText(overlay,
                        f"EAR: {s_ear:.2f}  Blinks: {self.blink_total}",
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
            # --------------------------------------------------
            # Display Head Pose Information
            # --------------------------------------------------
            #
            # Example:
            # Pitch: +5.2  Yaw: -12.4  Roll: +1.0
            #
            cv2.putText(overlay,
                        f"Pitch: {s_pitch:+.1f}  Yaw: {s_yaw:+.1f}  Roll: {roll:+.1f}",
                        (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (180, 180, 180), 1)

            # --------------------------------------------------
            # EAR Progress Bar
            # --------------------------------------------------
            #
            # Convert EAR value into a bar length.
            #
            # EAR = 0.40 → full bar
            # EAR = 0.20 → half bar
            # EAR = 0.00 → empty bar
            #
            bar_w = int(np.clip(s_ear / 0.40, 0, 1) * 200)
            cv2.rectangle(overlay, (20, 118), (220, 128), (50, 50, 50), -1) # Draw background bar
            cv2.rectangle(overlay, (20, 118), (20 + bar_w, 128), color, -1)# Draw filled EAR bar


        else:
            self.last_status = "No Face Detected"     # Update status
            self.blink_counter = 0     # Reset blink counter
            # Display warning text
            cv2.putText(overlay, "No Face Detected", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (100, 100, 255), 2)

        # --------------------------------------------------
        # Convert OpenCV Image Back To Video Frame
        # --------------------------------------------------
        #
        # OpenCV Image (NumPy Array)
        #            ↓
        #      AV VideoFrame
        #            ↓
        # Display in Streamlit
        #
        return av.VideoFrame.from_ndarray(overlay, format="bgr24")


# ── Sidebar ───────────────────────────────────────────────────────────────────
# --------------------------------------------------
# Sidebar Settings Panel
# --------------------------------------------------
# Everything inside this block appears in the
# Streamlit sidebar (left side of the app).
with st.sidebar:
    # Display sidebar title
    st.header("Settings")
    # --------------------------------------------------
    # Yaw Threshold Slider
    # --------------------------------------------------
    # Controls how much the user can turn their head
    # left or right before being marked as Distracted.
    #
    # Parameters:
    #   10 = minimum value
    #   40 = maximum value
    #   20 = default value
    #   1  = step size (changes by 1 each move)
    #
    yaw_th   = st.slider("Yaw threshold",       10, 40, 20, 1,
                          help="Left/right head turn angle to trigger Distracted")
     # --------------------------------------------------
    # Pitch UP Threshold Slider
    # --------------------------------------------------
    # Controls how much the user can look upward
    # before being considered distracted.
    #
    pitch_up = st.slider("Pitch UP threshold",   5, 35, 20, 1)
    # --------------------------------------------------
    # Pitch DOWN Threshold Slider
    # --------------------------------------------------
    # Controls how far downward the user can look.
    #
    # Reading or taking notes is allowed,
    # but looking sharply downward may indicate
    # distraction or drowsiness.
    #
    pitch_dn = st.slider("Pitch DOWN threshold", -50, -10, -20, 1,
                          help="Reading tilt is allowed; only sharp down triggers Distracted")
    # --------------------------------------------------
    # Drowsy Frame Count Slider
    # --------------------------------------------------
    # Number of consecutive frames with closed eyes
    # before the system labels the user as Drowsy.
    #
    # Example:
    # At ~30 FPS,
    # 8 frames ≈ 0.27 seconds
    #
    drowsy_f = st.slider("Drowsy frame count",  5, 40, 8, 1,
                         help="Frames of closed eyes before Drowsy (~30fps)")
    # Horizontal separator line
    st.markdown("---")
    # --------------------------------------------------
    # Live Statistics Section
    # --------------------------------------------------
    st.markdown("### Live Stats")
    # Empty placeholder for current status
    # Examples:
    # Focused, Distracted, Drowsy, No Face
    status_box = st.empty()
    # Empty placeholder for Eye Aspect Ratio (EAR)
    ear_box    = st.empty()
    # Empty placeholder for Pitch and Yaw values
    pose_box   = st.empty()
    # Empty placeholder for blink count
    blink_box  = st.empty()

# ── Stream ────────────────────────────────────────────────────────────────────
# --------------------------------------------------
# Start Webcam Stream
# --------------------------------------------------
# webrtc_streamer captures webcam frames and sends
# them to FocusProcessor for real-time analysis.
#
# Webcam
#    ↓
# FocusProcessor
#    ↓
# EAR, Head Pose, Blink Detection
#    ↓
# Display Results
#
ctx = webrtc_streamer(
    # Unique ID for this webcam component
    key="focus-tracker",
    # Send webcam frames and receive processed frames
    mode=WebRtcMode.SENDRECV,
    # Use FocusProcessor to process every frame
    video_processor_factory=FocusProcessor,
    # WebRTC connection configuration
    # Google's public STUN server helps establish
    # browser-to-browser communication.
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    # Webcam settings
    media_stream_constraints={
        # Request 640x480 video resolution
        # Lower resolution = faster processing
        "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},  # lower res = faster
        "audio": False # Disable microphone input
    },
    # Process frames immediately.
    # Prevents frame queue buildup and lag.
    async_processing=False,  # KEY: no queue buildup
    video_html_attrs={
        "style": {"width": "360px", "height": "270px", "margin": "0 auto", "display": "block"},
        "autoPlay": True,
        "controls": False,
    },
)
# --------------------------------------------------
# Update Processor Settings & Display Live Stats
# --------------------------------------------------

# Check if the webcam and processor are running
if ctx.video_processor:
    # Create a shorter reference to the processor
    proc = ctx.video_processor
    # --------------------------------------------------
    # Update thresholds using sidebar slider values
    # --------------------------------------------------

    # Head left/right distraction threshold
    proc.YAW_TH        = yaw_th
    # Looking upward threshold
    proc.PITCH_UP_TH   = pitch_up
    # Looking downward threshold
    proc.PITCH_DOWN_TH = pitch_dn
    # Number of closed-eye frames before drowsy
    proc.DROWSY_FRAMES = drowsy_f

    # --------------------------------------------------
    # Display Live Statistics
    # --------------------------------------------------

    # Current status:
    # Examples:
    # Focused, Distracted, Drowsy, No Face
    status_box.markdown(f"**Status:** `{proc.last_status}`")
    # Current Eye Aspect Ratio (EAR)
    # .3f = show 3 decimal places
    ear_box.markdown(f"**EAR:** `{proc.last_ear:.3f}`")
    # Current head pose values
    # Pitch = looking up/down
    # Yaw   = looking left/right
    #
    # +.1f means:
    # show sign (+/-)
    # show 1 decimal place
    pose_box.markdown(f"**Pitch:** `{proc.last_pitch:+.1f}`  **Yaw:** `{proc.last_yaw:+.1f}`")
     # Total number of detected blinks
    blink_box.markdown(f"**Blinks:** `{proc.blink_total}`")

# ── Legend ────────────────────────────────────────────────────────────────────
# --------------------------------------------------
# Status Legend Section
# --------------------------------------------------
# Add a horizontal line separator
# to visually separate this section
# from the webcam and statistics area.
st.markdown("---")
# Create 4 equal-width columns
# to display status meanings side-by-side.
c1, c2, c3, c4 = st.columns(4)

# --------------------------------------------------
# Focused Status
# --------------------------------------------------
# Green success box
# Indicates the student is attentive,
# looking at the screen, and eyes are open.
c1.success("Focused")
# --------------------------------------------------
# Drowsy Status
# --------------------------------------------------
# Yellow warning box
# Indicates eyes have remained closed
# for multiple consecutive frames.
c2.warning("Drowsy — sustained eye closure")
# --------------------------------------------------
# Distracted Status
# --------------------------------------------------
# Red error box
# Indicates the student's head is turned
# too far left/right or away from the screen.
c3.error("Distracted — head turned away")
# --------------------------------------------------
# Blinking Status
# --------------------------------------------------
# Blue informational box
# Normal blinking should not be considered
# drowsiness or distraction.
c4.info("Blinking / brief closure = normal")

# --------------------------------------------------
# User Guidance / Notes
# --------------------------------------------------
# Small text displayed below the legend.
#
# Explains that:
# - Slight downward head tilt for reading is okay.
# - Only large head turns are considered distraction.
# - Thresholds can be adjusted in the sidebar.
#
st.caption(
    "Reading posture (slight chin-down) is allowed. "
    "Only sharp head turns trigger Distracted. Tune thresholds in sidebar."
)