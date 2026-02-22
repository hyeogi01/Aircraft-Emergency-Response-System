import cv2, time, os, threading, collections, queue
import numpy as np
import face_recognition
from ultralytics import YOLO
from mediapipe.python.solutions import face_mesh
import logging
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# =========================================
# MediaPipe Face Mesh 준비
# =========================================
mp_face_mesh = face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
MOUTH_VERT = [13, 14]
MOUTH_HORI = [78, 308]

def euclidean(p, q):
    return np.linalg.norm(np.array(p) - np.array(q))

def eye_aspect_ratio(landmarks, eye_idx, img_w, img_h):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_idx]
    def to_xy(nl): return (nl.x*img_w, nl.y*img_h)
    P1, P2, P3, P4, P5, P6 = map(to_xy, [p1,p2,p3,p4,p5,p6])
    return (euclidean(P2,P6)+euclidean(P3,P5))/(2.0*euclidean(P1,P4))

# =========================================
# FrameReader (카메라 캡처 스레드)
# =========================================
class FrameReader(threading.Thread):
    def __init__(self, device=0, w=1280, h=720, mirror=True):
        super().__init__()
        self.cap = cv2.VideoCapture(device)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,h)
        self.q = queue.Queue(maxsize=1)
        self.running = True
        self.mirror = mirror

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                if self.mirror:
                    frame = cv2.flip(frame, 1)
                if not self.q.empty():
                    try: self.q.get_nowait()
                    except: pass
                self.q.put(frame)

    def read(self):
        if not self.q.empty():
            return self.q.get()
        return None

    def stop(self):
        self.running=False
        self.cap.release()

# =========================================
# Detector (YOLO + Face Recognition)
# =========================================
class Detector(threading.Thread):
    def __init__(self, yolo_model, known_encodings, known_names, mirror=True):
        super().__init__()
        self.yolo = YOLO(yolo_model)
        try: self.yolo.to('cuda:0')
        except: self.yolo.to('cpu')
        self.known_encodings = known_encodings
        self.known_names = known_names
        self.frame = None
        self.results = []
        self.face_state_dict = {}
        self.lock = threading.Lock()
        self.running = True
        self.ENC_INTERVAL = 30
        self.last_enc_time = {}
        self.frame_count = 0
        self.NORMAL_WINDOW = 5
        self.votes = {}
        self.mirror = mirror

    def set_frame(self, frame):
        self.frame = frame

    def run(self):
        while self.running:
            if self.frame is None:
                time.sleep(0.001)
                continue
            frame = self.frame.copy()
            h,w = frame.shape[:2]
            face_boxes=[]
            face_imgs=[]
            results = self.yolo.predict(frame, classes=[0], device='cuda:0', stream=True, verbose=False)
            for r in results:
                for box in r.boxes.xyxy:
                    x1,y1,x2,y2 = map(int, box)
                    x1,y1 = max(0,x1), max(0,y1)
                    x2,y2 = min(w-1,x2), min(h-1,y2)
                    if x2>x1 and y2>y1:
                        face_boxes.append((x1,y1,x2,y2))
                        face_imgs.append(frame[y1:y2,x1:x2])

            labels, states = [], []
            for idx, face_img in enumerate(face_imgs):
                face_id = idx
                need_enc = (face_id not in self.last_enc_time) or \
                           ((self.frame_count - self.last_enc_time[face_id])>=self.ENC_INTERVAL)
                label, state = "normal", "OK"
                # label,state = "abnormal","abnormal"

                # if need_enc:
                #     face_for_enc = cv2.flip(face_img, 1) if self.mirror else face_img
                #     rgb_face = cv2.cvtColor(face_for_enc, cv2.COLOR_BGR2RGB)
                #     encs = face_recognition.face_encodings(rgb_face)
                #     if encs:
                #         enc = encs[0]
                #         matches = face_recognition.compare_faces(
                #             self.known_encodings, enc, tolerance=0.5
                #         )
                #         if any(matches):
                #             label,state="normal","OK"
                #     self.last_enc_time[face_id]=self.frame_count
                # else:
                #     label,state = self.face_state_dict.get(face_id,("abnormal","abnormal"))

                if face_id not in self.votes:
                    self.votes[face_id]=collections.deque(maxlen=self.NORMAL_WINDOW)
                self.votes[face_id].append(1 if label=="normal" else 0)

                if sum(self.votes[face_id]) >= (self.NORMAL_WINDOW//2 + 1):
                    label,state="normal","OK"
                else:
                    label,state="abnormal","abnormal"

                labels.append(label)
                states.append(state)
                self.face_state_dict[face_id]=(label,state)

            with self.lock:
                self.results = list(zip(face_boxes,labels,states))
            self.frame_count+=1

# =========================================
# Landmark Processor (졸음 탐지)
# =========================================
class LandmarkProcessor(threading.Thread):
    def __init__(self):
        super().__init__()
        self.face_state_dict = {}
        self.face_boxes = []
        self.frame = None
        self.lock = threading.Lock()
        self.running=True
        self.ear_below_dict = {}
        self.EAR_THRESH=0.21
        self.PERCLOS_WINDOW=150

    def set_data(self, frame, face_results):
        self.frame=frame
        self.face_boxes = [r[0] for r in face_results]
        self.face_state_dict = {idx:(r[1],r[2]) for idx,r in enumerate(face_results)}

    def run(self):
        while self.running:
            if self.frame is None or not self.face_boxes:
                time.sleep(0.001)
                continue
            for idx,(x1,y1,x2,y2) in enumerate(self.face_boxes):
                label,state = self.face_state_dict[idx]
                if label=="normal":
                    face_roi = self.frame[y1:y2,x1:x2]
                    rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                    res = mp_face_mesh.process(rgb)
                    if res.multi_face_landmarks:
                        lms=res.multi_face_landmarks[0].landmark
                        ear_l=eye_aspect_ratio(lms,LEFT_EYE,x2-x1,y2-y1)
                        ear_r=eye_aspect_ratio(lms,RIGHT_EYE,x2-x1,y2-y1)
                        ear=(ear_l+ear_r)/2
                        is_closed=1 if ear<self.EAR_THRESH else 0
                        if idx not in self.ear_below_dict:
                            self.ear_below_dict[idx]=collections.deque(maxlen=self.PERCLOS_WINDOW)
                        self.ear_below_dict[idx].append(is_closed)
                        perclos=sum(self.ear_below_dict[idx])/len(self.ear_below_dict[idx])
                        state="DROWSY" if perclos>0.35 else "OK"
                        self.face_state_dict[idx]=(label,state)
            time.sleep(0.001)

# =========================================
# 크로마키 처리 함수
# =========================================
background = cv2.imread("image.png")
if background is None:
    raise FileNotFoundError("image.png not found!")

def apply_chroma(frame):
    background_resized = cv2.resize(background, (frame.shape[1], frame.shape[0]))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 80, 80])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)
    fg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    bg = cv2.bitwise_and(background_resized, background_resized, mask=mask)
    return cv2.add(fg, bg)

# =========================================
# 실행 함수: run_cam_multithread
# =========================================
def run_cam_multithread(facebank_dir='facebank', device=0, yolo_model='yolov11n.pt', mirror=True,
                        shared=None):
    enc_path = os.path.join(facebank_dir, 'encodings.npy')
    names_path = os.path.join(facebank_dir, 'names.npy')
    if not (os.path.exists(enc_path) and os.path.exists(names_path)):
        raise FileNotFoundError("Facebank not found.")
    known_encodings = np.load(enc_path, allow_pickle=True)
    known_names = np.load(names_path, allow_pickle=True)

    reader = FrameReader(device, mirror=mirror)
    reader.start()
    detector = Detector(yolo_model, known_encodings, known_names, mirror=mirror)
    detector.start()
    landmark = LandmarkProcessor()
    landmark.start()

    fps, tick, f = 0.0, time.time(), 0

    try:
        while True:
            frame = reader.read()
            if frame is None:
                time.sleep(0.001)
                continue

            detector.set_frame(frame)
            with detector.lock:
                face_results = detector.results.copy()
            landmark.set_data(frame, face_results)

            for idx, (box, label, state) in enumerate(face_results):
                x1, y1, x2, y2 = box
                l, state = landmark.face_state_dict.get(idx, (label, state))
                color = (255,0,0) if state=="OK" else (0,0,255) if state=="DROWSY" else (0,165,255)
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(frame, state, (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            f += 1
            if f % 10 == 0:
                now = time.time()
                fps = 10 / (now - tick)
                tick = now
            # cv2.putText(frame, f"FPS:{fps:.1f}", (10,65),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            # ✅ 크로마키 적용
            frame = apply_chroma(frame)

            # ✅ 최신 프레임을 서버 쪽으로 공유
            if shared is not None:
                with shared["lock"]:
                    shared["camera_frame"] = frame.copy()

    finally:
        reader.stop()
        detector.running = False
        landmark.running = False
        cv2.destroyAllWindows()
