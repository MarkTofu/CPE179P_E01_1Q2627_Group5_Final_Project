"""Webcam client. Run on WINDOWS (WSL can't see the webcam). SPACE = capture, Q/Esc = quit."""
import cv2
import requests

URL = "http://localhost:8000/predict"

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    raise SystemExit("Cannot open webcam (is another app using it?)")

msg = "SPACE = capture, Q = quit"
while True:
    ok, frame = cam.read()
    if not ok:
        raise SystemExit("Webcam read failed")
    show = frame.copy()
    cv2.putText(show, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("PesoSight", show)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), 27):
        break
    if key == ord(" "):
        jpg = cv2.imencode(".jpg", frame)[1].tobytes()
        try:
            r = requests.post(URL, files={"file": ("capture.jpg", jpg, "image/jpeg")}, timeout=10)
            r.raise_for_status()
            d = r.json()
            msg = (f"PHP {d['denomination']} ({d['confidence']:.0%})" if d["denomination"]
                   else f"Not sure ({d['confidence']:.0%}) - try again")
        except requests.RequestException:
            msg = "Server error - is uvicorn running?"
        print(msg)

cam.release()
cv2.destroyAllWindows()
