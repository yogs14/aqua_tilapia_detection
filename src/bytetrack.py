import supervision as sv

class ByteTracker:
    def __init__(self, track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=30):
        # Inisialisasi ByteTrack dari library supervision
        self.sv_tracker = sv.ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=match_thresh,
            frame_rate=frame_rate
        )

    def update(self, results):
        """
        Menerima hasil raw dari YOLO (Ultralytics), melakukan tracking,
        dan mengembalikan list objek dengan format [x1, y1, x2, y2, id].
        """
        # 1. Konversi hasil YOLO ke format Supervision
        detections = sv.Detections.from_ultralytics(results)

        # 2. Update tracker
        detections = self.sv_tracker.update_with_detections(detections)

        # 3. Format output agar bisa dibaca oleh VideoProcessor
        # Return format: list of [x1, y1, x2, y2, track_id]
        tracked_objects = []
        
        if detections.tracker_id is not None:
            for box, track_id in zip(detections.xyxy, detections.tracker_id):
                x1, y1, x2, y2 = box
                # Tambahkan ke list
                tracked_objects.append([x1, y1, x2, y2, int(track_id)])
        
        return tracked_objects