
<div align="center">
  <h1>Gesture Volume Control</h1>
 </div>

> Uses OpenCV and MediaPipe to Control system volume 

## 💾 REQUIREMENTS
+ opencv-python
+ mediapipe
+ comtypes
+ numpy
+ pycaw

```bash
pip install -r requirements.txt
```
***

## 📝 CODE EXPLANATION
<b>Importing Libraries</b>
```py
import cv2
import mediapipe as mp
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
```
***
Solution API's 
```py
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
```
***

Volume Control Library Usage 
```py
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
```
***
Getting Volume Range using `volume.GetVolumeRange()` Method
```py
volRange = volume.GetVolumeRange()
minVol , maxVol , volBar, volPer= volRange[0] , volRange[1], 400, 0
```
***
Setting up webCam using OpenCV
```py
wCam, hCam = 640, 480
cam = cv2.VideoCapture(0)
cam.set(3,wCam)
cam.set(4,hCam)
```
***
Using MediaPipe Hand Landmark Model for identifying Hands 
```py
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

  while cam.isOpened():
    success, image = cam.read()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
            )
```
***
Using multi_hand_landmarks method for Finding postion of Hand landmarks
```py
lmList = []
    if results.multi_hand_landmarks:
      myHand = results.multi_hand_landmarks[0]
      for id, lm in enumerate(myHand.landmark):
        h, w, c = image.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        lmList.append([id, cx, cy])    
```
***
Assigning variables for Thumb and Index finger position
```py
if len(lmList) != 0:
      x1, y1 = lmList[4][1], lmList[4][2]
      x2, y2 = lmList[8][1], lmList[8][2]
```
***
Marking Thumb and Index finger using `cv2.circle()` and Drawing a line between them using `cv2.line()`
```py
cv2.circle(image, (x1,y1),15,(255,255,255))  
cv2.circle(image, (x2,y2),15,(255,255,255))  
cv2.line(image,(x1,y1),(x2,y2),(0,255,0),3)
length = math.hypot(x2-x1,y2-y1)
if length < 50:
    cv2.line(image,(x1,y1),(x2,y2),(0,0,255),3)
```
***
Converting Length range into Volume range using `numpy.interp()`
```py
vol = np.interp(length, [50, 220], [minVol, maxVol])
```
***
Changing System Volume using `volume.SetMasterVolumeLevel()` method
```py
volume.SetMasterVolumeLevel(vol, None)
volBar = np.interp(length, [50, 220], [400, 150])
volPer = np.interp(length, [50, 220], [0, 100])
```
***
Drawing Volume Bar using `cv2.rectangle()` method
```py
cv2.rectangle(image, (50, 150), (85, 400), (0, 0, 0), 3)
cv2.rectangle(image, (50, int(volBar)), (85, 400), (0, 0, 0), cv2.FILLED)
cv2.putText(image, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
        1, (0, 0, 0), 3)}

```
***
Displaying Output using `cv2.imshow` method
```py
cv2.imshow('handDetector', image) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
```
***
Closing webCam
```py
cam.release()
```
***

<div align = "center">
