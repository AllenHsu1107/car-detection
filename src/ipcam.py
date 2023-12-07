import cv2

# Replace 'your_ip_address' and 'your_port' with the actual IP address and port of your camera
camera_url = "http://192.168.1.197:4747/video"

# Open the video stream
cap = cv2.VideoCapture(camera_url)

pixelx = 1920
pixely = 1080

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    frame = cv2.resize(frame,(pixelx,pixely))

    # Display the frame
    cv2.imshow("IP Camera Feed", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close the window
cap.release()
cv2.destroyAllWindows()
