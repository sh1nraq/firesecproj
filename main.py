import cv2
from config import Config
from fire_detector import Detector


def main():
    try:
        detector = Detector(Config.MODEL_PATH, iou_threshold=0.20)

        cap = cv2.VideoCapture(str(Config.VIDEO_SOURCE))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, detection = detector.process_frame(frame)

            cv2.imshow("Fire Detection System", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
