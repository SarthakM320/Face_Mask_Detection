import cv2
import time
from tensorflow.keras.models import load_model
import numpy as np

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model('mask_detection.h5')


def detect_and_predict_mask(frame):
    bboxes = classifier.detectMultiScale(frame)
    img_array = []
    for box in bboxes:
        x, y, width, height = box
        x2, y2 = x + width, y + height
        img_array.append(frame[x:x2, y:y2])
    preds = []
    for img in img_array:
        img = cv2.resize(img, (224, 224))
        pred = np.argmax(model.predict(img.reshape(1, 224, 224, 3)), axis=-1)
        preds.append(pred)

    for pred, bbox in zip(preds, bboxes):
        x, y, width, height = bbox
        x2, y2 = x + width, y + height
        if pred == 0:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        x = int(x)
        x2 = int(x2)
        y = int(y)
        y2 = int(y2)
        frame = cv2.rectangle(frame, (x, y), (x2, y2), color, 2)

    return frame


vs = cv2.VideoCapture(0)
time.sleep(2.0)

while True:
    ret, frame = vs.read()
    print(frame)
    # if frame is None:
    img = detect_and_predict_mask(frame)
    cv2.imshow("", img)
    # else:
    #     cv2.imshow("", frame)
    #     pass

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # time.sleep(1.0)

vs.release()
cv2.destroyAllWindows()
