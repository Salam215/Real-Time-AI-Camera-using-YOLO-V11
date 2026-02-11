import cv2
from ultralytics import YOLO
import threading

# Load the model
yolo = YOLO("C:\\Users\\Inspection_P611\\anaconda3\\envs\\AI\\Project\\Models_Used\\PH6_Model.pt")
lock = threading.Lock()
stop_flag = False

def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] * 
    (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)
def get_coordinate(rst):
    for box in rst.boxes:
            [x1, y1, x2, y2] = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    return(x1, y1, x2, y2)

def capture_frames():
    videoCap = cv2.VideoCapture("rtsp://admin:admin@192.168.65.15/profile?token=media_profile1&TO=60")
    while not stop_flag:
        global frame, results
        ret, frame = videoCap.read()
        if not ret:
            continue
    videoCap.release()

def display_results():
    global frame, results, stop_flag
    while not stop_flag:
        with lock:
                results = yolo.track(frame, stream=True)
                for rst in results:
                        classes_names = rst.names
                        for box in rst.boxes:
                            if box.conf[0] > 0.4:
                                x1, y1, x2 , y2 = get_coordinate(rst)
                                cls = int(box.cls[0])
                                color = getColours(cls)
                                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                                cv2.putText(frame, f'{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.imshow("AI Camera",frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_flag = True
#Start Thread
if __name__ == '__main__':
    threads = [threading.Thread(target = capture_frames),
           threading.Timer(2, display_results)
]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

cv2.destroyAllWindows()