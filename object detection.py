import cv2
from ultralytics import YOLO
def detect_webcam(model_name='yolov8n.pt'):
    model = YOLO(model_name)
    cap  = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow('YOLOv8 Detection', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
def main():
        detect_webcam()
if __name__ =='__main__':
        main()
    
        
        