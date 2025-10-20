#Python version 3.10.10
#Numpy version 1.26.4
#OpenCV version 4.10.0.84
#MediaPiPe version 0.10.21

import mediapipe as mp 
import cv2
import math
import sys 
import io

# проверяем кодировку
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# получаем точки рук
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils 

def classify_gesture(hand_landmarks):
    h, w = 1, 1
    
    def distance(p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    def is_finger_straight(finger_tip, finger_pip, finger_mcp, threshold=0.9):
        a = distance(finger_mcp, finger_pip)
        b = distance(finger_pip, finger_tip)
        c = distance(finger_mcp, finger_tip)
        if a == 0 or b == 0:
            return False
        angle = math.degrees(math.acos((a*a + b*b - c*c) / (2*a*b)))
        return angle > threshold *180
    
    def get_finger_state():
        states = []
        
        thumb_tip = hand_landmarks.landmark[4]
        thumb_ip = hand_landmarks.landmark[3]
        thumb_mcp = hand_landmarks.landmark[2]
        thumb_is_straight = is_finger_straight(thumb_tip, thumb_ip, thumb_mcp, 0.7)
        states.append(thumb_is_straight)
        
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        finger_mcps = [5, 9, 13, 17]
        
        for tip, pip, mcp in zip(finger_tips, finger_pips, finger_mcps):
            tip_lm = hand_landmarks.landmark[tip]
            pip_lm = hand_landmarks.landmark[pip]
            mcp_lm = hand_landmarks.landmark[mcp]
            straight = is_finger_straight(tip_lm, pip_lm, mcp_lm)
            states.append(straight)
            
        return states
    
    fingers = get_finger_state()
    
    wrist = hand_landmarks.landmark[0]
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    index_mcp = hand_landmarks.landmark[5]

    # проверка на жест указательного пальца ввурх
    if (fingers[1] and not any(fingers[2:])):
        if index_tip.y < index_mcp.y and index_tip.y < wrist.y:
            return 'Pointing Up'

    # проверка на жест ножниц
    elif not fingers[0] and fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
        if index_tip.y < wrist.y and middle_tip.y < wrist.y:
            return 'Scissors'

    # проверка на жест рока
    elif not fingers[0] and fingers[1] and not fingers[2] and not fingers[3] and fingers[4]:
        return 'Rock On'

    # проверка на жест камня
    elif not fingers[0] and not fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
        return 'Rock'

    # проверка на жест бумаги
    elif all(fingers):
        return 'Paper'

    # проверка на жест пальца вверх/вниз
    elif (fingers[0] and not any(fingers[1:])):
        if thumb_tip.y < wrist.y:
            return 'Thumb up'
        elif thumb_tip.y > wrist.y:
            return 'Thumb down'
            
    # возвращаем неизвестно если не можем угадать
    else:
        return 'Unknown'

# создаём камеру для записи видео в реальном времени
cap = cv2.VideoCapture(0)

# настраиваем
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_gesture_video.mp4', fourcc, fps, (frame_width, frame_height))

# записываем видео
while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            gesture = classify_gesture(hand_landmarks)

            cv2.putText(image, f'Gesture: {gesture if gesture != None else "Unknown"}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Записываем кадр в файл
    out.write(image)

    cv2.imshow('Hand Gesture Recognition', image)
    if cv2.waitKey(5) & 0xFF == 27 or cv2.getWindowProperty('Hand Gesture Recognition', cv2.WND_PROP_VISIBLE) < 1:
        break

# сохраняем видео и закрываем камеру
cap.release()
out.release()

cv2.destroyAllWindows()
