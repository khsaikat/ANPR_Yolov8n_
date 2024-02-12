import time
from ultralytics import YOLO

from sort.sort import *
from utils import *

results = {}
# A simple online and realtime tracking algorithm for 2D multiple object tracking in video sequences.
sort_tracker = Sort()

# Pretrained YOLOv8n Model from Ultralytics
coco_yolo_v8_model = YOLO('./models/yolov8n.pt')
license_plate_detection_model = YOLO('./models/license_plate_detector.pt')

# Different Types of Vehicle Class
vehicles = [2, 3, 5, 7]

# WebCam Initialization
capture = cv2.VideoCapture("./test_media/plate_track_recog.mp4")  # "./test_media/plate_track_recog.mp4"

# Visualize the frames of the capture, Live
frame_number = -1
ret = True

while ret:
    frame_number += 1
    ret, frame = capture.read()

    # Drop every other frame
    # if frame_number % 2 == 0:
    #     continue
    if ret:
        # Resize the video frame to 720p
        # frame = cv2.resize(frame, (1280, 720))

        # Closing the video by Escape button
        if cv2.waitKey(10) == 27:
            break

        results[frame_number] = {}

        # Taking every 60th frame
        # if frame_number % 10 == 0:
        # Detected Object
        detections = coco_yolo_v8_model(frame)[0]
        vehicles_detected = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection

            # Taking only Vehicle classes
            if int(class_id) in vehicles:
                vehicles_detected.append([x1, y1, x2, y2, score])

        # Tracking the moving Vehicles
        tracking_ids = sort_tracker.update(np.asarray(vehicles_detected))
        # print(tracking_ids)
        # for tracking_id in tracking_ids:
        #     x1, y1, x2, y2, class_id = tracking_id
        #     draw_border(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Detect Licence plates
        license_plate_detections = license_plate_detection_model(frame)[0]
        for license_plate in license_plate_detections.boxes.data.tolist():
            # print(license_plate)
            x1, y1, x2, y2, score, class_id = license_plate
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)

            # Vehicle and License plate combination
            v_x1, v_y1, v_x2, v_y2, v_id = get_car(license_plate, tracking_ids)
            draw_border(frame, (int(v_x1), int(v_y1)), (int(v_x2), int(v_y2)), (0, 255, 0), 2)

            if v_id != -1:
                # Taking only the license plate from the frame
                license_plate_cropped = frame[int(y1):int(y2), int(x1): int(x2), :]

                # Processing the cropped license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_cropped, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_threshold = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                license_plate_text, license_plate_text_score = ocr_license_plate(license_plate_crop_threshold)

                if license_plate_text is not None:

                    H, W, _ = license_plate_cropped.shape

                    try:
                        frame[int(v_y1) - H - 10:int(v_y1) - 10,
                                int((v_x2 + v_x1 - W) / 2):int((v_x2 + v_x1 + W) / 2), :] = license_plate_cropped
                        frame[int(v_y1) - H - 50:int(v_y1) - H - 10,
                                int((v_x2 + v_x1 - W) / 2):int((v_x2 + v_x1 + W) / 2), :] = (255, 255, 255)
                        (text_width, text_height), _ = cv2.getTextSize(license_plate_text, cv2.FONT_HERSHEY_SIMPLEX,
                                                                       2, 5)
                        cv2.putText(frame, license_plate_text, (
                            int((v_x2 + v_x1 - text_width) / 2), int(v_y1 - H - 25 + (text_height / 2))),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    2,
                                    (0, 0, 0),
                                    5)

                    except:
                        pass

                    results[frame_number][v_id] = {'car': {'bbox': [v_x1, v_y1, v_x2, v_y2]},
                                                   'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                     'text': license_plate_text,
                                                                     'bbox_score': score,
                                                                     'text_score': license_plate_text_score}}

        # Saving every 100th frames
        # if frame_number % 100 == 0:
        #     cv2.imwrite('./test_media/webcam_' + str(frame_number) + '.jpg', frame)
        #     print("webcam_" + str(frame_number))

        # resized_ = cv2.resize(frame, (1280, 720))
        frame = cv2.resize(frame, (1280, 720))
        cv2.imshow('frame', frame)

# Write initial results in a CSV file
write_csv(results, './results/test' + str(time.time()) + '.csv')

capture.release()
cv2.destroyAllWindows()
