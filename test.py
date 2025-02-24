import streamlit as st 
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose



cap = cv2.VideoCapture(0)

bt_stop = st.button('STOP')
em = st.empty()
with mp_hands.Hands(min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
# , mp_pose.Pose(
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results_hands = hands.process(image)
        # results_pose = pose.process(image)
        image.flags.writeable = True
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw hand landmarks
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Draw pose landmarks
        # if results_pose.pose_landmarks:
        #     mp_drawing.draw_landmarks(
        #         image, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # cv2.imshow('MediaPipe Hands and Pose', image)
        em.image(image, channels="RGB")
        if bt_stop:
            break

cap.release()
cv2.destroyAllWindows()