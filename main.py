from threading import Thread
import cv2
from ultralytics import YOLO

class DoorbellModel():
    def __init__(self):
        self.model = YOLO("./best.pt")

        self.frame = None
        self.smart_frame = None

        self.stopped = True
        self.t = Thread(target=self.update)
        self.t.daemon = True

    def start(self):
        self.stopped = False
        self.t.start()

    def stop(self):
        self.stopped = True

    def update(self):
        while self.stopped is False:
            if self.frame is None:
                continue

            results = self.model.predict(source=self.frame,conf=0.1)
            result = results[0]

            self.smart_frame = result.plot(labels=True)

def main():
    video_capture = cv2.VideoCapture(0)
    doorbell_model = DoorbellModel()
    doorbell_model.start()

    video_capturing = True
    while video_capturing:
        ret, frame = video_capture.read()

        if ret:
            doorbell_model.frame = frame

        smart_frame = doorbell_model.smart_frame
        if smart_frame is not None:
            cv2.imshow("webcam", smart_frame)

        key_hex = cv2.waitKey(1) & 0xFF

        if key_hex == ord("q"):
            video_capturing = False

    doorbell_model.stop()
    video_capture.release()
    cv2.destroyAllWindows()

    
if __name__ == "__main__":
    main()