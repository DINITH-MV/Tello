from tello import *
import tello as _tello
import cv2
import threading
import time
import numpy as np

# --- State ---
running = True
latest_qr = ""
is_flying = False

rc_left_right = 0
rc_forward_back = 0
rc_up_down = 0
rc_yaw = 0

SPEED = 60  # RC speed scale (0-100)

# --- Virtual joystick state ---
# Each stick: (x, y) in [-1, 1]
_left_stick  = [0.0, 0.0]   # axis 0=LR, 1=FB
_right_stick = [0.0, 0.0]   # axis 0=YAW, 1=UD
_dragging_left  = False
_dragging_right = False

# --- Shared video/QR state (written by qr thread, read by main thread) ---
_latest_frame = None
_display_qr_text = ""
_display_qr_points = None
_display_qr_until = 0.0
_frame_lock = threading.Lock()

JOY_WIN  = "Tello Virtual Joystick"
JOY_W, JOY_H = 520, 260
JOY_R = 90        # stick circle radius
KNOB_R = 18       # knob radius
LEFT_CX,  LEFT_CY  = 130, 130
RIGHT_CX, RIGHT_CY = 390, 130


# ---------------------------------------------------------------------------
# Virtual joystick helpers
# ---------------------------------------------------------------------------

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _joy_mouse(event, x, y, flags, param):
    global _left_stick, _right_stick, _dragging_left, _dragging_right

    def norm(cx, cy):
        dx = _clamp((x - cx) / JOY_R, -1.0, 1.0)
        dy = _clamp((y - cy) / JOY_R, -1.0, 1.0)
        # Keep knob inside circle
        mag = (dx**2 + dy**2) ** 0.5
        if mag > 1.0:
            dx, dy = dx / mag, dy / mag
        return [dx, dy]

    if event == cv2.EVENT_LBUTTONDOWN:
        if (x - LEFT_CX)**2 + (y - LEFT_CY)**2 <= JOY_R**2:
            _dragging_left = True
        elif (x - RIGHT_CX)**2 + (y - RIGHT_CY)**2 <= JOY_R**2:
            _dragging_right = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if _dragging_left:
            _left_stick = norm(LEFT_CX, LEFT_CY)
        if _dragging_right:
            _right_stick = norm(RIGHT_CX, RIGHT_CY)

    elif event == cv2.EVENT_LBUTTONUP:
        if _dragging_left:
            _left_stick = [0.0, 0.0]
            _dragging_left = False
        if _dragging_right:
            _right_stick = [0.0, 0.0]
            _dragging_right = False


def _draw_joystick_panel():
    panel = np.zeros((JOY_H, JOY_W, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)

    def draw_stick(cx, cy, stick, label_top, label_bot, label_left, label_right):
        # Outer ring
        cv2.circle(panel, (cx, cy), JOY_R, (80, 80, 80), 2)
        # Crosshair
        cv2.line(panel, (cx - JOY_R, cy), (cx + JOY_R, cy), (60, 60, 60), 1)
        cv2.line(panel, (cx, cy - JOY_R), (cx, cy + JOY_R), (60, 60, 60), 1)
        # Knob
        kx = int(cx + stick[0] * JOY_R)
        ky = int(cy + stick[1] * JOY_R)
        cv2.circle(panel, (kx, ky), KNOB_R, (0, 180, 255), -1)
        cv2.circle(panel, (kx, ky), KNOB_R, (0, 220, 255), 1)
        # Labels
        cv2.putText(panel, label_top,   (cx - 20, cy - JOY_R - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160,160,160), 1)
        cv2.putText(panel, label_bot,   (cx - 20, cy + JOY_R + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160,160,160), 1)
        cv2.putText(panel, label_left,  (cx - JOY_R - 38, cy + 4),  cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160,160,160), 1)
        cv2.putText(panel, label_right, (cx + JOY_R + 6,  cy + 4),  cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160,160,160), 1)

    draw_stick(LEFT_CX,  LEFT_CY,  _left_stick,
               "Fwd", "Back", "Left", "Right")
    draw_stick(RIGHT_CX, RIGHT_CY, _right_stick,
               "Up", "Down", "CCW", "CW")

    # Buttons
    btn_color_takeoff = (0, 180, 0)  if not is_flying else (40, 40, 40)
    btn_color_land    = (0, 0, 200)  if is_flying     else (40, 40, 40)
    cv2.rectangle(panel, (10, 210), (120, 248), btn_color_takeoff, -1)
    cv2.putText(panel, "TAKEOFF", (18, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.rectangle(panel, (140, 210), (240, 248), btn_color_land, -1)
    cv2.putText(panel, "LAND",    (158, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.rectangle(panel, (260, 210), (380, 248), (0, 0, 140), -1)
    cv2.putText(panel, "EMERGENCY", (263, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,80,80), 1)

    # Status
    status = "FLYING" if is_flying else "LANDED"
    cv2.putText(panel, "Status: " + status, (390, 228),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255) if is_flying else (120,120,120), 1)

    return panel


def _btn_click(event, x, y, flags, param):
    global is_flying, running
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if 210 <= y <= 248:
        if 10 <= x <= 120 and not is_flying:
            print("Takeoff!")
            try:
                takeoff()
                is_flying = True
            except Exception as e:
                print("Takeoff error:", e)
        elif 140 <= x <= 240 and is_flying:
            print("Landing!")
            try:
                land()
                is_flying = False
            except Exception as e:
                print("Land error:", e)
        elif 260 <= x <= 380:
            print("Emergency stop!")
            _tello._send("emergency")
            is_flying = False
            running = False


def _combined_mouse(event, x, y, flags, param):
    _joy_mouse(event, x, y, flags, param)
    _btn_click(event, x, y, flags, param)


# ---------------------------------------------------------------------------


def draw_qr_box(frame, points, qr_text):
    if points is None:
        return frame
    points = points[0].astype(int)
    for i in range(len(points)):
        pt1 = tuple(points[i])
        pt2 = tuple(points[(i + 1) % len(points)])
        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
    x, y = points[0]
    cv2.putText(frame, "QR: " + qr_text, (int(x), int(y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame


def qr_scan_thread():
    """Background thread: grabs frames, runs QR detection, stores results.
    Never calls cv2.imshow — all rendering happens on the main thread."""
    global running, latest_qr
    global _latest_frame, _display_qr_text, _display_qr_points, _display_qr_until

    qr_detector = cv2.QRCodeDetector()
    last_print_time = 0

    print("QR scanner thread started...")

    while running:
        frame = get_video_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        frame = cv2.resize(frame, (640, 480))
        current_time = time.time()

        qr_text, points, _ = qr_detector.detectAndDecode(frame)
        if qr_text:
            latest_qr = qr_text
            _display_qr_text = qr_text
            _display_qr_points = points
            _display_qr_until = current_time + 1.0
            if current_time - last_print_time > 1:
                print("QR Code detected:", qr_text)
                last_print_time = current_time

        with _frame_lock:
            _latest_frame = frame


def main_loop():
    """Main thread: renders joystick panel + video window, handles all cv2 GUI."""
    global running, rc_left_right, rc_forward_back, rc_up_down, rc_yaw

    VIDEO_WIN = "Tello QR Control"

    cv2.namedWindow(JOY_WIN)
    cv2.setMouseCallback(JOY_WIN, _combined_mouse)

    print("Virtual joystick window opened.")
    print("  Left stick  - Forward/Back + Left/Right")
    print("  Right stick - Up/Down + Yaw")
    print("  TAKEOFF / LAND / EMERGENCY buttons on the panel")
    print("  Press Q to quit")

    last_rc_time = 0
    video_win_open = False

    while running:
        # --- Joystick panel ---
        panel = _draw_joystick_panel()
        cv2.imshow(JOY_WIN, panel)

        # --- RC values from sticks ---
        rc_left_right   = int(_left_stick[0]  * SPEED)
        rc_forward_back = int(-_left_stick[1] * SPEED)
        rc_up_down      = int(-_right_stick[1] * SPEED)
        rc_yaw          = int(_right_stick[0]  * SPEED)

        now = time.time()
        if is_flying and now - last_rc_time >= 0.05:   # 20 Hz
            _tello._send("rc %d %d %d %d" % (rc_left_right, rc_forward_back, rc_up_down, rc_yaw))
            last_rc_time = now

        # --- Video + QR window (only when frames are available) ---
        with _frame_lock:
            frame = _latest_frame.copy() if _latest_frame is not None else None

        if frame is not None:
            # Draw QR box if still within display window
            if _display_qr_text and _display_qr_points is not None and now < _display_qr_until:
                frame = draw_qr_box(frame, _display_qr_points, _display_qr_text)

            # HUD
            status = "FLYING" if is_flying else "LANDED"
            cv2.putText(frame, "Status: " + status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            cv2.putText(frame,
                        "LR:%d FB:%d UD:%d YAW:%d" % (rc_left_right, rc_forward_back, rc_up_down, rc_yaw),
                        (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            if latest_qr:
                cv2.putText(frame, "Last QR: " + latest_qr[:40], (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1)

            if not video_win_open:
                cv2.namedWindow(VIDEO_WIN)
                video_win_open = True
            cv2.imshow(VIDEO_WIN, frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            running = False
            break

    cv2.destroyAllWindows()


def drone_init():
    """Connect to drone, start video, and launch QR thread. Runs in background."""
    global running
    try:
        print("Connecting to drone...")
        start()
        power = get_battery()
        print("Battery: %d%%" % power)

        if power < 20:
            print("Battery too low for safe flight.")
            return

        start_video()
        time.sleep(3)

        qr_thread = threading.Thread(target=qr_scan_thread, daemon=True)
        qr_thread.start()

    except Exception as e:
        print("Drone init error:", e)
        print("Running in offline mode — joystick UI only.")


# Start drone connection in background so the joystick UI opens immediately
init_thread = threading.Thread(target=drone_init, daemon=True)
init_thread.start()

try:
    # All OpenCV GUI runs on the main thread
    main_loop()

except KeyboardInterrupt:
    print("Stopped by user.")
    try:
        if is_flying:
            land()
    except Exception:
        pass

except Exception as e:
    print("Error:", e)
    try:
        if is_flying:
            land()
    except Exception:
        pass

finally:
    running = False
    try:
        stop_video()
    except Exception:
        pass
    print("Last QR detected:", latest_qr)
    print("Program finished.")
