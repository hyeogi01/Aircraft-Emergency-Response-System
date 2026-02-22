import os, cv2, time, collections, numpy as np
import face_recognition
from ultralytics import YOLO
from mediapipe.python.solutions import face_mesh

# ===== MediaPipe Face Mesh 준비 =====
mp_face_mesh = face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
MOUTH_VERT = [13, 14]
MOUTH_HORI = [78, 308]


def euclidean(p, q):
    return np.linalg.norm(np.array(p) - np.array(q))


def eye_aspect_ratio(landmarks, eye_idx, img_w, img_h):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_idx]

    def to_xy(nl): return (nl.x * img_w, nl.y * img_h)

    P1, P2, P3, P4, P5, P6 = map(to_xy, [p1, p2, p3, p4, p5, p6])
    return (euclidean(P2, P6) + euclidean(P3, P5)) / (2.0 * euclidean(P1, P4))


def mouth_aspect_ratio(landmarks, mv_idx, mh_idx, img_w, img_h):
    u, d = [landmarks[i] for i in mv_idx]
    l, r = [landmarks[i] for i in mh_idx]
    U, D = (u.x * img_w, u.y * img_h), (d.x * img_w, d.y * img_h)
    L, R = (l.x * img_w, l.y * img_h), (r.x * img_w, r.y * img_h)
    return euclidean(U, D) / (euclidean(L, R) + 1e-6)


# ===== 카메라 열기 =====
def open_cam(dev, w=1280, h=720, fps=30):
    cap = cv2.VideoCapture(dev)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {dev}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Cannot read frame from {dev}")
    return cap


# ===== 안정화된 실시간 실행 (최적화) =====
def run_cam_optimized(facebank_dir='facebank', device=0, yolo_model='yolo11n.pt', skip_frame=2):
    enc_path = os.path.join(facebank_dir, 'encodings.npy')
    names_path = os.path.join(facebank_dir, 'names.npy')
    if not (os.path.exists(enc_path) and os.path.exists(names_path)):
        raise FileNotFoundError("Facebank not found. Run build_facebank first.")

    known_encodings = np.load(enc_path, allow_pickle=True)
    known_names = np.load(names_path, allow_pickle=True)

    # YOLO 모델
    yolo = YOLO(yolo_model)
    try:
        yolo.to('cuda:0')  # GPU 사용 가능 시
    except:
        yolo.to('cpu')
    cap = open_cam(device)

    EAR_THRESH = 0.21
    YAWN_THRESH = 0.65
    CLOSED_MIN_FRAMES = 8
    PERCLOS_WINDOW = 150
    ear_below_dict = {}  # 얼굴 ID별 PERCLOS 기록
    fps, tick, f = 0.0, time.time(), 0
    prev_faces, prev_states, prev_labels = [], [], []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        do_detection = (f % skip_frame == 0) or (f == 0)
        face_imgs, face_boxes = [], []
        if do_detection:
            results = yolo.predict(frame, classes=[0], device='cuda:0', stream=False)
            for r in results:
                for box in r.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    face_imgs.append(frame[y1:y2, x1:x2])
                    face_boxes.append((x1, y1, x2, y2))

            states, labels = [], []
            for face_img in face_imgs:
                rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                encs = face_recognition.face_encodings(rgb_face)
                if encs:
                    enc = encs[0]
                    matches = face_recognition.compare_faces(known_encodings, enc, tolerance=0.4)
                    if any(matches):
                        label = "normal"
                        state = "OK"
                    else:
                        label = "abnormal"
                        state = "abnormal"
                else:
                    label = "abnormal"
                    state = "abnormal"
                labels.append(label)
                states.append(state)
            prev_faces, prev_states, prev_labels = face_boxes, states, labels
        else:
            face_boxes, states, labels = prev_faces, prev_states, prev_labels

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_face_mesh.process(rgb)

        for idx, (x1, y1, x2, y2) in enumerate(face_boxes):
            label = labels[idx]
            state = states[idx]
            # 인가된 사람만 EAR/MAR/PERCLOS 계산
            if label == "normal" and res.multi_face_landmarks:
                lms = res.multi_face_landmarks[0].landmark
                ear_l = eye_aspect_ratio(lms, LEFT_EYE, w, h)
                ear_r = eye_aspect_ratio(lms, RIGHT_EYE, w, h)
                ear = (ear_l + ear_r) / 2
                mar = mouth_aspect_ratio(lms, MOUTH_VERT, MOUTH_HORI, w, h)
                is_closed = 1 if ear < EAR_THRESH else 0

                face_id = idx
                if face_id not in ear_below_dict:
                    ear_below_dict[face_id] = collections.deque(maxlen=PERCLOS_WINDOW)
                ear_below_dict[face_id].append(is_closed)
                perclos = sum(ear_below_dict[face_id]) / len(ear_below_dict[face_id])
                if perclos > 0.35:
                    state = "DROWSY"
                else:
                    state = "OK"

            # 박스 및 상태 표시
            color = (0, 200, 0) if state == "OK" else (0, 0, 255) if state == "DROWSY" else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, state, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        f += 1
        if f % 10 == 0:
            now = time.time()
            fps = 10 / (now - tick)
            tick = now

        cv2.putText(frame, f"FPS:{fps:.1f}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Face & Drowsiness", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    device = 0
    run_cam_optimized(device=device)