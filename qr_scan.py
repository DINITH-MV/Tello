from tello import *
import cv2
import threading
import time
import numpy as np

running = True
latest_qr = ""


def draw_qr_box(frame, points, qr_text):
    if points is None:
        return frame

    points = points[0].astype(int)

    for i in range(len(points)):
        pt1 = tuple(points[i])
        pt2 = tuple(points[(i + 1) % len(points)])
        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    x, y = points[0]
    cv2.putText(
        frame,
        "QR: " + qr_text,
        (int(x), int(y) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    return frame


def qr_scanner():
    global running, latest_qr

    qr_detector = cv2.QRCodeDetector()

    last_print_time = 0
    last_qr_text = ""
    last_points = None
    last_detection_time = 0

    print("QR scanner started...")

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
            last_qr_text = qr_text
            last_points = points
            last_detection_time = current_time

            if current_time - last_print_time > 1:
                print("QR Code detected:", qr_text)
                last_print_time = current_time

        # Show QR box for only 1 second
        if last_qr_text and last_points is not None and current_time - last_detection_time < 1:
            frame = draw_qr_box(frame, last_points, last_qr_text)

        cv2.imshow("Tello QR Scanner", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            running = False
            break

    cv2.destroyAllWindows()


try:
    start()

    power = get_battery()
    print("Power Level: " + str(power) + "%")

    if power < 20:
        print("Battery is too low for safe takeoff.")
        running = False
    else:
        # Start Tello video using your tello.py function
        start_video()
        time.sleep(3)

        # Start QR scanning in background
        qr_thread = threading.Thread(target=qr_scanner)
        qr_thread.daemon = True
        qr_thread.start()

        time.sleep(2)

        up(30)
        forward(100)
        down(30)
        time.sleep(2)

except KeyboardInterrupt:
    print("Stopped by user.")
    try:
        land()
    except Exception:
        pass

except Exception as e:
    print("Error:", e)
    try:
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