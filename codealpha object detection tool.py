import torch
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
tracker = DeepSort(max_age=30)


cap = cv2.VideoCapture(0)  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()

    tracks = []
    for *box, conf, cls in detections:
        x1, y1, x2, y2 = box
        tracks.append(([x1, y1, x2 - x1, y2 - y1], conf, int(cls)))

    track_ids = tracker.update_tracks(tracks, frame=frame)

    for track in track_ids:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow('Real-time Object Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
