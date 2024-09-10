from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np

app = Flask(__name__)

# Load reference image and extract face encoding
reference_image_path = 'image.jpg'  # Replace with your reference image path
reference_image = face_recognition.load_image_file(reference_image_path)
reference_encoding = face_recognition.face_encodings(reference_image)[0]

# Initialize video capture
cap = cv2.VideoCapture(0)  # Change 0 to video file path if needed

def generate_frames():
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert image to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Find all face locations and encodings in the image
        face_locations = face_recognition.face_locations(rgb_image)
        current_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        for (top, right, bottom, left), encoding in zip(face_locations, current_encodings):
            # Compare the encodings with the reference encoding
            matches = face_recognition.compare_faces([reference_encoding], encoding)
            name = "Ichrak" if any(matches) else "Unknown"

            # Draw a box around the face
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0) if name == "Ichrak" else (0, 0, 255), 2)

            # Draw a label with the name
            cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12) if name == "Ichrak" else (0, 0, 255), 2)

        # Encode the image to JPEG format
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()

        # Yield the frame in the byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
