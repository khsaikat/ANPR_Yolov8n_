import cv2

# Define a video capture object
vid_cap = cv2.VideoCapture("./plate_track_recog.mp4")

# Capture video frame by frame
success, image = vid_cap.read()

# Declare the variable with value 0
count = 0

# Creating a loop for running the video
# and saving all the frames
while success:
    # Capture video frame by frame
    success, image = vid_cap.read()

    # Resize the image frames
    resize = cv2.resize(image, (1280, 720))

    # Saving the frames with certain names
    cv2.imwrite("%04d.jpg" % count, resize)

    # Closing the video by Escape button
    if cv2.waitKey(10) == 27:
        break

    # Incrementing the variable value by 1
    count += 1
