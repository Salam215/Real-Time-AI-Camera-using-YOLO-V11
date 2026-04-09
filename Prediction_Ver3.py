stop_flag = False

def getColours(cls_num):
    match cls_num:
         case 0:
              return ((0,0,255))
         case 1:
              return ((0,255,0)) 
    #return tuple(color)
def get_coordinate(rst):
    for box in rst.boxes:
            [x1, y1, x2, y2] = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    return(x1, y1, x2, y2)
def Read_Frames(RSTP, frame_queue): #Producer
    try:
        import cv2
        global stop_flag    
        cap = cv2.VideoCapture(RSTP)
        if not cap.isOpened():
            print("Camera Error")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Stream error")
                break
            if not frame_queue.full():
                frame_queue.put(frame)
            else:
                pass
        cap.release()
    except KeyboardInterrupt:
        pass
def Detection_Process(frame_camera, model_path,flag_bool): #Consumer

    try:
        from ultralytics import YOLO, solutions
        from collections import defaultdict
        import cv2,time
        time.sleep(1)
        model = YOLO(model_path)
        global stop_flag, track_ids
        track_ids = []
        # class_list = model.names
        # line = [(900,400), (500, 400)]
        class_counts = {'Broken-Bag': 0, 'Normal' : 0}
        crossed_id = []
        while not stop_flag:
            if not frame_camera.empty():
                frame = frame_camera.get()
                results = model.track(frame, persist= True)
                if results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                for rst in results:
                    classes_names = rst.names
                    for box in rst.boxes:
                        if box.conf[0] > 0.6:
                            x1, y1, x2 , y2 = get_coordinate(rst)
                            cls = int(box.cls[0])
                            color = getColours(cls)
                            cx = (x1 + x2) // 2
                            cy = (y1 + y2) // 2
                            #cv2.circle(frame, (cx, cy),4, (0,0,255), -1)
                            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                            cv2.putText(frame, f'{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                            if classes_names[cls] == 'Broken-Bag':
                                flag_bool.value = 1
                            for track_id in track_ids:
                                if cy < 550:
                                    if track_id not in crossed_id:
                                        detected = track_ids.pop()
                                        crossed_id.append(detected)
                                        class_counts[classes_names[cls]] = class_counts.get(classes_names[cls]) + 1 
                print(track_ids)
                lineheight = 0
                for key, value in class_counts.items():
                    cv2.putText(frame, f"{key}: {value}", (200, 200 + lineheight), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    lineheight += 35
                #cv2.line(frame, (900, 550), (500, 550), (0,0,255), 3)
                cv2.namedWindow("AI Camera", cv2.WINDOW_NORMAL)
                cv2.setWindowProperty("AI Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow("AI Camera", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_flag = True
                frame_camera.task_done()
        cv2.destroyAllWindows()
    except KeyboardInterrupt:
        pass
def PLC_Connection(input):
    try:
        import snap7
        from snap7.util import set_bool
        import time
        global stop_flag
        #PLC Connection
        IP = '192.168.11.15'
        Rack = 0
        Slot = 1
        plc = snap7.client.Client()
        plc.connect(IP, Rack, Slot)
        PLC_Data = bytearray(1)
        while True:
            if bool(input.value):
                time.sleep(2)
                if bool(input.value):
                    set_bool(PLC_Data, 0,0,1)
                    plc.write_area(snap7.type.Areas.MK, 0, 0 , PLC_Data)
                    input.value = 0
                else:
                    continue
    except KeyboardInterrupt:
        pass
if __name__ == '__main__':
   try: 
    from multiprocessing import Process
    import multiprocessing as mp
    import time
    RSTP_URL = "rtsp://admin:InstrumentP611@192.168.11.61:554/Streaming/channels/101"
    YOLO_Model = "C:\\Users\\Inspection_P611\\anaconda3\\envs\\AI\\Project\\Models_Used\\Epoch53_PH6Model.pt"
    frame_queue = mp.JoinableQueue(maxsize=1)
    flag = mp.Value('b', 0)

    Reader = Process(target= Read_Frames, args=(RSTP_URL, frame_queue))
    Detector = Process(target=Detection_Process, args=(frame_queue, YOLO_Model,flag))
    PLCKoneksi = Process(target=PLC_Connection, args= (flag,))

    Reader.start()
    Detector.start()
    PLCKoneksi.start()

    Detector.join()
    PLCKoneksi.join()
   except KeyboardInterrupt:
       pass
