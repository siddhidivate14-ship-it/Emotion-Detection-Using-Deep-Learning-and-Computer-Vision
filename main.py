import cv2
from fer import Video, FER
import matplotlib.pyplot as plt


# Initialize the emotion detector
detector = FER(mtcnn=True)



# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect emotion in the frame
    emotion, score = detector.top_emotion(frame)
    
    # Display the detected emotion on screen
    if emotion:
        cv2.putText(frame, f"{emotion}: {score:.2f}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the video window
    cv2.imshow('Emotion Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
