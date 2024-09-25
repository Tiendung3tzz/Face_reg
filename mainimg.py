from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import cvlib as cv
# Load model
model = load_model('gender_detection4')

classes = ['man', 'woman']

# Đường dẫn tới ảnh đầu vào
image_path = 'datatest/hary.jpeg'

# Đọc ảnh từ đường dẫn
frame = cv2.imread(image_path)
width, height, _ = frame.shape

if width > 1700 or width < 400:
    new_size = (1280, 750)
    frame = cv2.resize(frame, new_size)



# Apply face detection
face, confidence = cv.detect_face(frame)

for idx, f in enumerate(face):
    startX, startY, endX, endY = f

    # Draw rectangle over face
    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Crop the detected face region
    face_crop = np.copy(frame[startY:endY, startX:endX])
    if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
        continue

    # Preprocessing for gender detection model
    face_crop = cv2.resize(face_crop, (96, 96))
    face_crop = face_crop.astype("float") / 255.0
    face_crop = img_to_array(face_crop)
    face_crop = np.expand_dims(face_crop, axis=0)

    # Apply gender detection on face
    conf = model.predict(face_crop)[0]
    idx = np.argmax(conf)
    label = classes[idx]
    label = "{}".format(label)

    Y = startY - 10 if startY - 10 > 10 else startY + 10

    # Write label and confidence above face rectangle
    cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Hiển thị ảnh đầu ra
cv2.imshow("gender detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
