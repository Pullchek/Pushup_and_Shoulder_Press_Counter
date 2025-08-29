
import cv2
import mediapipe as mp
import numpy as np
import math
import time

# Initialize MediaPipe solutions
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Initialize variables
pushup_counter = 0
pushup_stage = "up"
shoulder_press_counter = 0
shoulder_press_stage = "down"
start_time = time.time()
calories_burned = 0
feedback = "Position yourself for exercise"
exercise_mode = "pushup"  # Default exercise mode

# Constants
PUSHUP_CALORIES = 0.5  # Approximate calories burned per pushup
SHOULDER_PRESS_CALORIES = 0.4  # Approximate calories burned per shoulder press
TARGET_REPS = 10  # Target number of repetitions


# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point

    # Calculate the angle
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    # Ensure the angle is within 0-180 degrees
    if angle > 180.0:
        angle = 360 - angle

    return angle


# Function to draw a professional dashboard
def draw_dashboard(image, exercise_mode, pushup_counter, pushup_stage, shoulder_press_counter,
                   shoulder_press_stage, elapsed_time, calories, feedback, arm_angle=None, shoulder_angle=None):
    height, width, _ = image.shape

    # Create semi-transparent overlay for dashboard
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (width, 120), (50, 50, 50), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

    # Draw divider lines
    cv2.line(image, (width // 3, 0), (width // 3, 120), (200, 200, 200), 1)
    cv2.line(image, (2 * width // 3, 0), (2 * width // 3, 120), (200, 200, 200), 1)

    # Display current exercise mode
    cv2.putText(image, "EXERCISE MODE", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, exercise_mode.upper(), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)

    # Display repetition counts based on exercise mode
    if exercise_mode == "pushup":
        counter = pushup_counter
        stage = pushup_stage
    else:
        counter = shoulder_press_counter
        stage = shoulder_press_stage

    cv2.putText(image, f"REPS: {counter}/{TARGET_REPS}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1,
                cv2.LINE_AA)

    # Display current stage
    cv2.putText(image, "STAGE", (width // 3 + 20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    # Different color based on stage
    stage_color = (0, 255, 0) if stage == "up" else (0, 165, 255)
    cv2.putText(image, stage.upper(), (width // 3 + 30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, stage_color, 1, cv2.LINE_AA)

    # Display form feedback
    cv2.putText(image, "FORM", (width // 3 + 20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Display relevant angle for current exercise
    if exercise_mode == "pushup" and arm_angle is not None:
        cv2.putText(image, f"Arm angle: {arm_angle:.1f}°", (width // 3 + 20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
    elif exercise_mode == "shoulder press" and shoulder_angle is not None:
        cv2.putText(image, f"Shoulder angle: {shoulder_angle:.1f}°", (width // 3 + 20, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Display statistics (time and calories)
    cv2.putText(image, "STATISTICS", (2 * width // 3 + 20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA)
    minutes, seconds = divmod(int(elapsed_time), 60)
    cv2.putText(image, f"Time: {minutes:02d}:{seconds:02d}", (2 * width // 3 + 20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, f"Calories: {calories:.1f}", (2 * width // 3 + 20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, f"Total reps: {pushup_counter + shoulder_press_counter}", (2 * width // 3 + 20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Display feedback at bottom of screen
    feedback_bg = image.copy()
    cv2.rectangle(feedback_bg, (0, height - 40), (width, height), (50, 50, 50), -1)
    cv2.addWeighted(feedback_bg, 0.7, image, 0.3, 0, image)
    cv2.putText(image, feedback, (20, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Progress bar for repetitions
    progress_percent = min(counter / TARGET_REPS, 1.0)
    bar_width = int(width * 0.8)
    bar_start = (width - bar_width) // 2
    bar_height = 15
    cv2.rectangle(image, (bar_start, height - 65), (bar_start + bar_width, height - 50), (100, 100, 100), -1)
    cv2.rectangle(image, (bar_start, height - 65), (bar_start + int(bar_width * progress_percent), height - 50),
                  (0, 255, 0), -1)

    # Instructions for changing exercise mode
    cv2.putText(image, "Press 'p' for Pushup mode | Press 's' for Shoulder Press mode",
                (20, height - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    return image


# Initialize webcam
cap = cv2.VideoCapture(0)

# Set up pose detection
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        # Read frame
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture video")
            break

        # Calculate elapsed time and calories
        elapsed_time = time.time() - start_time
        calories_burned = (pushup_counter * PUSHUP_CALORIES) + (shoulder_press_counter * SHOULDER_PRESS_CALORIES)

        # Convert to RGB and process
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make pose detection
        results = pose.process(image)

        # Convert back to BGR for display
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Variables to store current angles
        arm_angle = None
        shoulder_angle = None

        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates for pushup tracking
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculate arm angle for pushups
            arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

            # Get coordinates for shoulder press tracking
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]

            # Calculate shoulder angle for shoulder press
            shoulder_angle = calculate_angle(left_hip, left_shoulder, left_elbow)

            # Exercise detection logic based on current mode
            if exercise_mode == "pushup":
                # Pushup counter logic
                if arm_angle > 160:
                    pushup_stage = "down"
                    feedback = "Good form! Now push up"
                if arm_angle < 90 and pushup_stage == "down":
                    pushup_stage = "up"
                    pushup_counter += 1
                    feedback = f"Great job! Completed {pushup_counter} pushups"
                    if pushup_counter == TARGET_REPS:
                        feedback = "Congratulations! You've reached your pushup target!"
            else:  # shoulder press mode
                # Shoulder press counter logic
                if shoulder_angle > 150:  # Arms extended upward
                    shoulder_press_stage = "up"
                    feedback = "Good! Now lower your arms"
                if shoulder_angle < 90 and shoulder_press_stage == "up":
                    shoulder_press_stage = "down"
                    shoulder_press_counter += 1
                    feedback = f"Great job! Completed {shoulder_press_counter} shoulder presses"
                    if shoulder_press_counter == TARGET_REPS:
                        feedback = "Congratulations! You've reached your shoulder press target!"

        except Exception as e:
            feedback = "Position yourself properly for detection"
            pass

        # Draw pose landmarks with improved styling
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing_styles.get_default_pose_landmarks_style()
            )

        # Apply dashboard
        image = draw_dashboard(
            image, exercise_mode, pushup_counter, pushup_stage,
            shoulder_press_counter, shoulder_press_stage,
            elapsed_time, calories_burned, feedback, arm_angle, shoulder_angle
        )

        # Display the resulting frame
        cv2.imshow('Exercise Detection System', image)

        # Handle key presses
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):  # Exit application
            break
        elif key == ord('p'):  # Switch to pushup mode
            exercise_mode = "pushup"
            feedback = "Switched to Pushup mode"
        elif key == ord('s'):  # Switch to shoulder press mode
            exercise_mode = "shoulder press"
            feedback = "Switched to Shoulder Press mode"
        elif key == ord('r'):  # Reset counters
            pushup_counter = 0
            shoulder_press_counter = 0
            feedback = "Counters reset"
            start_time = time.time()

# Release resources
cap.release()
cv2.destroyAllWindows()