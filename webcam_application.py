import main
import cv2

test= main.k_means(3)
cap= test.cap

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    segmented_image,masked_image = test.k_means_filtering(frame)  # process the image in every frame and show to screen
    cv2.imshow('segmented image', segmented_image)
    cv2.imshow('masked image', masked_image)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindow