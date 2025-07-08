#!/usr/bin/env python3
"""
Multi-Exercise QT Robot Coach - Heart Rate and HRV Monitoring System
Based on sympathetic_feedback_rospy.py template with 5 exercise options
Integrates with Polar H10 sensor data and QT Robot for real-time feedback
"""

import time
import random
from threading import Thread, Lock
import rospy
from std_msgs.msg import String
import pandas as pd
import simpleaudio
import os

# User configuration
AGE = 35
MAX_THRESH = .80
UPPER_THIRD = 0.75
LOWER_THIRD = 0.65
MIN_THRESH = .40

max_hr = 220 - AGE

# Exercise-specific configurations
EXERCISE_CONFIG = {
    "squats": {
        "name": "Squats",
        "instructions": "Stand with feet shoulder-width apart, lower your body as if sitting back into a chair, then return to standing.",
        "metronome_normal": 1.2,  # slower pace for squats
        "metronome_fast": 0.8,
        "metronome_slow": 1.8,
        "intensity_modifier": 0.85,  # slightly lower intensity
        "duration": 60,
        "motivational_phrases": [
            "Keep your chest up and core engaged!",
            "Remember to push through your heels!",
            "Great form! Keep those knees behind your toes!",
            "You're building strong legs!"
        ]
    },
    "lunges": {
        "name": "Lunges", 
        "instructions": "Step forward with one leg, lowering your hips until both knees are bent at 90 degrees. Alternate legs.",
        "metronome_normal": 1.0,
        "metronome_fast": 0.7,
        "metronome_slow": 1.5,
        "intensity_modifier": 0.90,
        "duration": 60,
        "motivational_phrases": [
            "Keep your torso upright!",
            "Feel the burn in those legs!",
            "Perfect form! Switch legs when ready!",
            "You're getting stronger with each rep!"
        ]
    },
    "high_knees": {
        "name": "High Knees",
        "instructions": "Run in place, bringing your knees up toward your chest with each step.",
        "metronome_normal": 0.4,  # faster pace for high knees
        "metronome_fast": 0.25,
        "metronome_slow": 0.6,
        "intensity_modifier": 1.15,  # higher intensity
        "duration": 60,
        "motivational_phrases": [
            "Get those knees up high!",
            "Pump those arms!",
            "Feel the cardio burn!",
            "You're flying! Keep it up!"
        ]
    },
    "push_ups": {
        "name": "Push-Ups",
        "instructions": "Start in plank position, lower your body until chest nearly touches floor, then push back up.",
        "metronome_normal": 1.5,  # slower for strength exercise
        "metronome_fast": 1.0,
        "metronome_slow": 2.0,
        "intensity_modifier": 0.75,  # lower cardio intensity
        "duration": 60,
        "motivational_phrases": [
            "Keep that core tight!",
            "Lower all the way down!",
            "Strong push! Feel those arms working!",
            "You're building upper body strength!"
        ]
    },
    "burpees": {
        "name": "Burpees",
        "instructions": "Start standing, drop to squat, jump back to plank, do push-up, jump feet to squat, then jump up with arms overhead.",
        "metronome_normal": 2.0,  # slower for complex movement
        "metronome_fast": 1.5,
        "metronome_slow": 3.0,
        "intensity_modifier": 1.25,  # highest intensity
        "duration": 45,  # shorter duration due to intensity
        "motivational_phrases": [
            "Full body power! You've got this!",
            "Each burpee makes you stronger!",
            "Feel that total body burn!",
            "You're a warrior! Keep fighting!"
        ]
    }
}

# Thread-safe state management
class ExerciseState:
    def __init__(self):
        self.lock = Lock()
        self._rest_hr = None
        self._inst_hr = -1
        self._mean_hr = -1
        self._rmssd_inc = False
        self._done = False
        self._current_exercise = None
        
        # Exercise state
        self.warm_up_start = True
        self.slow_down = False
        self.inc_speed = False
        self.first_time = True
        self.stop_metronome = False
        self.exercise_selected = False
        self.baseline_complete = False
    
    @property
    def rest_hr(self):
        with self.lock:
            return self._rest_hr
    
    @rest_hr.setter
    def rest_hr(self, value):
        with self.lock:
            self._rest_hr = value
    
    @property
    def mean_hr(self):
        with self.lock:
            return self._mean_hr
    
    @mean_hr.setter
    def mean_hr(self, value):
        with self.lock:
            self._mean_hr = value
    
    @property
    def done(self):
        with self.lock:
            return self._done
    
    @done.setter
    def done(self, value):
        with self.lock:
            self._done = value
    
    @property
    def current_exercise(self):
        with self.lock:
            return self._current_exercise
    
    @current_exercise.setter
    def current_exercise(self, value):
        with self.lock:
            self._current_exercise = value

# Global state
state = ExerciseState()

# ROS publishers - will be initialized in main
speech_say_pub = None
emotion_show_pub = None
gesture_play_pub = None
audio_play_pub = None

def safe_publish(publisher, message_data, topic_name="Unknown"):
    """Safely publish to ROS topic with error handling"""
    try:
        if publisher is not None:
            msg = String()
            msg.data = message_data
            publisher.publish(msg)
            print(f"[{topic_name}] {message_data}")
            return True
        else:
            print(f"[ERROR] {topic_name} publisher not initialized")
            return False
    except Exception as e:
        print(f"[ERROR] Failed to publish to {topic_name}: {e}")
        return False

def get_exercise_thresholds(exercise_name):
    """Calculate exercise-specific heart rate thresholds"""
    if state.rest_hr is None:
        return None
    
    config = EXERCISE_CONFIG.get(exercise_name, EXERCISE_CONFIG["squats"])
    modifier = config["intensity_modifier"]
    
    hrr = (max_hr - state.rest_hr) * modifier
    
    return {
        "min_thresh": (MIN_THRESH * hrr) + state.rest_hr,
        "max_thresh": (MAX_THRESH * hrr) + state.rest_hr,
        "upper_third": (UPPER_THIRD * hrr) + state.rest_hr,
        "lower_third": (LOWER_THIRD * hrr) + state.rest_hr
    }

def select_exercise():
    """Handle exercise selection process"""
    print("Starting exercise selection...")
    
    # Present exercise options
    exercises = list(EXERCISE_CONFIG.keys())
    
    message = "Great! Now let's choose your exercise. I have five options for you:"
    safe_publish(speech_say_pub, message, "Speech")
    time.sleep(3)
    
    for i, exercise_key in enumerate(exercises, 1):
        exercise_name = EXERCISE_CONFIG[exercise_key]["name"]
        message = f"Option {i}: {exercise_name}"
        safe_publish(speech_say_pub, message, "Speech")
        time.sleep(2)
    
    message = "Please tell me which number you'd like to choose, from 1 to 5."
    safe_publish(speech_say_pub, message, "Speech")
    
    # Simulate user selection (for demo - replace with actual input mechanism)
    print("Simulating exercise selection...")
    time.sleep(5)
    
    # Random selection for demo (replace with actual user input)
    selected_index = random.randint(0, len(exercises) - 1)
    selected_exercise = exercises[selected_index]
    
    exercise_name = EXERCISE_CONFIG[selected_exercise]["name"]
    instructions = EXERCISE_CONFIG[selected_exercise]["instructions"]
    
    message = f"Excellent choice! You selected {exercise_name}."
    safe_publish(speech_say_pub, message, "Speech")
    time.sleep(2)
    
    message = f"Here's how to do {exercise_name}: {instructions}"
    safe_publish(speech_say_pub, message, "Speech")
    safe_publish(gesture_play_pub, 'QT/happy', "Gesture")
    
    state.current_exercise = selected_exercise
    state.exercise_selected = True
    
    print(f"Selected exercise: {exercise_name}")
    return selected_exercise

def speak_feedback(warm_up, give_feedback, warm_up_count):
    """Provide feedback based on heart rate and current exercise"""
    current_rest_hr = state.rest_hr
    current_mean_hr = state.mean_hr
    current_exercise = state.current_exercise
    
    # Check if we have valid data
    if current_rest_hr is None or current_mean_hr == -1:
        print("Waiting for valid heart rate data...")
        return 0

    # Get exercise-specific thresholds
    thresholds = get_exercise_thresholds(current_exercise)
    if not thresholds:
        return 0

    min_thresh = thresholds["min_thresh"]
    max_thresh = thresholds["max_thresh"]
    upper_third = thresholds["upper_third"]
    lower_third = thresholds["lower_third"]

    # Exercise-specific motivational statements
    if current_exercise and current_exercise in EXERCISE_CONFIG:
        exercise_phrases = EXERCISE_CONFIG[current_exercise]["motivational_phrases"]
    else:
        exercise_phrases = ["Keep going!", "You're doing great!", "Stay strong!"]

    # General motivational statements
    mot_statements = [
        "You are doing good! Would you like to go faster?",
        "Keep pushing! Let me know if you want a faster pace!",
        "Hang in there, let me know if you want higher speed",
        "Keep going for some more time! You are doing great!"
    ]
    
    inc_statements = [
        "Let's increase the pace!", "Try to increase your pace!", 
        "If possible, increase your pace!"
    ]

    print(f"Feedback - Exercise: {current_exercise}, Warm-up: {warm_up}, HR: {current_mean_hr:.1f}")
    
    # Show emotion
    safe_publish(emotion_show_pub, 'QT/happy', "Emotion")
    
    if warm_up:
        print(f"Warm-up count: {warm_up_count}")

        if current_mean_hr <= current_rest_hr:
            if warm_up_count < 1:
                message = random.choice(inc_statements)
                safe_publish(speech_say_pub, message, "Speech")
                time.sleep(0.5)
                safe_publish(gesture_play_pub, 'QT/happy', "Gesture")
            return 0

        if current_rest_hr < current_mean_hr < max_thresh:
            if warm_up_count < 1:
                message = random.choice(exercise_phrases)
                safe_publish(speech_say_pub, message, "Speech")
                time.sleep(0.5)
                safe_publish(gesture_play_pub, 'QT/happy', "Gesture")
                time.sleep(0.5)
                safe_publish(emotion_show_pub, 'QT/happy', "Emotion")
            return 0

        if current_mean_hr > max_thresh:
            if warm_up_count < 1:
                message = 'Looks like you are ready to start the exercise!'
                safe_publish(speech_say_pub, message, "Speech")
                time.sleep(0.5)
                safe_publish(gesture_play_pub, 'QT/happy', "Gesture")
                time.sleep(0.5)
                safe_publish(emotion_show_pub, 'QT/happy', "Emotion")
            return 0

    if not warm_up:
        if current_mean_hr > max_thresh:
            if give_feedback == 10:
                message = 'Slow down and dont over exert yourself'
                safe_publish(speech_say_pub, message, "Speech")
                time.sleep(0.5)
                safe_publish(gesture_play_pub, 'QT/happy', "Gesture")
            return 0

        if current_mean_hr < min_thresh:
            if give_feedback < 1:
                exercise_name = EXERCISE_CONFIG[current_exercise]["name"]
                message = f'You are resting, lets continue with {exercise_name}!'
                safe_publish(speech_say_pub, message, "Speech")
                time.sleep(0.5)
                safe_publish(gesture_play_pub, 'QT/happy', "Gesture")
                print("You are resting")
            return 0

        if current_mean_hr > upper_third:
            if give_feedback < 1:
                if not state.inc_speed:
                    message = 'You are doing great, would you like to slow down?'
                    safe_publish(speech_say_pub, message, "Speech")
                    time.sleep(0.5)
                    safe_publish(gesture_play_pub, 'QT/happy', "Gesture")
                else:
                    message = random.choice(exercise_phrases)
                    safe_publish(speech_say_pub, message, "Speech")
                    time.sleep(0.5)
                    safe_publish(gesture_play_pub, 'QT/happy', "Gesture")

                # Simulate user response (30% chance to slow down)
                print("Simulating user input...")
                time.sleep(2)
                if random.random() < 0.3:
                    state.slow_down = True
                    print("Simulated: User chose to slow down")
                else:
                    print("Simulated: User chose not to slow down")
            return 0

        if current_mean_hr > min_thresh or current_mean_hr > lower_third:
            if give_feedback < 1:
                if not state.inc_speed:
                    message = random.choice(mot_statements)
                    safe_publish(speech_say_pub, message, "Speech")
                    time.sleep(0.5)
                    safe_publish(gesture_play_pub, 'QT/happy', "Gesture")
                else:
                    message = random.choice(exercise_phrases)
                    safe_publish(speech_say_pub, message, "Speech")
                    time.sleep(0.5)
                    safe_publish(gesture_play_pub, 'QT/happy', "Gesture")

                # Simulate user response (20% chance to increase speed)
                print("Simulating user input...")
                time.sleep(2)
                if random.random() < 0.2:
                    state.inc_speed = True
                    print("Simulated: User chose to increase speed")
                else:
                    print("Simulated: User chose not to increase speed")
            return 0

def calc_initial_hr():
    """Calculate baseline heart rate"""
    print("Calculating initial heart rate...")
    
    while state.mean_hr == -1:
        print("Waiting for heart rate data...")
        time.sleep(10)

    state.rest_hr = state.mean_hr
    state.baseline_complete = True
    
    print(f"Resting HR: {state.rest_hr:.1f}")
    print(f"Max HR: {max_hr}")
    print(f"Upper HR limit: {(MAX_THRESH * (max_hr - state.rest_hr)) + state.rest_hr:.1f}")
    print(f"Lower HR limit: {(MIN_THRESH * (max_hr - state.rest_hr)) + state.rest_hr:.1f}")

def play_metronome():
    """Play metronome with exercise-specific timing"""
    print("Starting metronome...")
    
    try:
        strong_beat = simpleaudio.WaveObject.from_wave_file('strong_beat.wav')
        silence = simpleaudio.WaveObject.from_wave_file('silence_1.wav')
    except Exception as e:
        print(f"Error loading audio files: {e}")
        return

    while not state.done and not rospy.is_shutdown():
        try:
            current_exercise = state.current_exercise
            
            if current_exercise and current_exercise in EXERCISE_CONFIG:
                config = EXERCISE_CONFIG[current_exercise]
                normal_tempo = config["metronome_normal"]
                fast_tempo = config["metronome_fast"] 
                slow_tempo = config["metronome_slow"]
            else:
                # Default tempos
                normal_tempo = 0.5
                fast_tempo = 0.4
                slow_tempo = 0.75

            if state.slow_down:
                strong_beat.play()
                time.sleep(slow_tempo)
            elif state.inc_speed:
                strong_beat.play()
                time.sleep(fast_tempo)
            elif state.stop_metronome:
                silence.play()
                time.sleep(0.5)
            elif not state.warm_up_start and state.exercise_selected:
                strong_beat.play()
                time.sleep(normal_tempo)
            else:
                time.sleep(0.5)
                
        except Exception as e:
            print(f"Metronome error: {e}")
            time.sleep(0.5)

def calc_feedback():
    """Main exercise feedback logic"""
    print("Starting exercise feedback system...")
    
    calc_initial_hr()
    
    message = "The baseline collection is complete. Before you start, make sure you have enough space around you."
    safe_publish(speech_say_pub, message, "Speech")
    time.sleep(5)

    # Exercise selection phase
    selected_exercise = select_exercise()
    exercise_config = EXERCISE_CONFIG[selected_exercise]
    
    time.sleep(3)

    # Exercise parameters
    training_phase_limit = exercise_config["duration"]  # Use exercise-specific duration
    give_feedback_limit = 12
    warm_up_time = 30
    warm_up_limit = 20
    give_feedback_count = give_feedback_limit
    warm_up_count = warm_up_limit

    # Start warm-up
    message = 'Lets start a warm-up! You can start with light movement to prepare for your exercise.'
    safe_publish(speech_say_pub, message, "Speech")
    safe_publish(gesture_play_pub, 'QT/happy', "Gesture")
    time.sleep(1)

    # Warm-up phase
    while state.warm_up_start and not rospy.is_shutdown():
        warm_up_count -= 1
        warm_up_time -= 1
        print("Warm-up phase...")
        
        if state.rest_hr is not None:
            thresholds = get_exercise_thresholds(selected_exercise)
            if thresholds:
                print(f"Exercise: {exercise_config['name']}")
                print(f"HR thresholds - Upper: {thresholds['upper_third']:.1f}, Lower: {thresholds['lower_third']:.1f}, Min: {thresholds['min_thresh']:.1f}")
                print(f"Current mean HR: {state.mean_hr:.1f}")

        speak_feedback(state.warm_up_start, 0, warm_up_count)
        
        if warm_up_count < 1:
            warm_up_count = warm_up_limit
        if warm_up_time < 1:
            state.warm_up_start = False
            
        time.sleep(1)

    # Main exercise phase
    training_phase = training_phase_limit
    state.first_time = True
    time.sleep(2)

    exercise_name = exercise_config["name"]
    message = f'Let us start {exercise_name}! I am playing a metronome to help with pacing.'
    safe_publish(speech_say_pub, message, "Speech")
    safe_publish(gesture_play_pub, 'QT/happy', "Gesture")
    time.sleep(5)

    # Exercise loop
    while not state.done and not rospy.is_shutdown():
        training_phase -= 1
        give_feedback_count -= 1

        if training_phase > 1:
            speak_feedback(False, give_feedback_count, 100)
            
        if state.rest_hr is not None:
            print(f"Exercise phase - Training: {training_phase}, Feedback: {give_feedback_count}")
            print(f"Current mean HR: {state.mean_hr:.1f}")

        if give_feedback_count < 1:
            give_feedback_count = give_feedback_limit

        # Rest break after first round
        if training_phase == 0 and state.first_time:
            state.first_time = False
            state.inc_speed = False
            state.slow_down = False
            state.stop_metronome = True
            
            message = f'Great work on {exercise_name}! Lets take thirty seconds to catch our breath!'
            safe_publish(speech_say_pub, message, "Speech")
            safe_publish(gesture_play_pub, 'QT/happy', "Gesture")
            time.sleep(30)

            message = 'If you want to end this session, let me know! Otherwise, we can go for a second round!'
            safe_publish(speech_say_pub, message, "Speech")
            safe_publish(gesture_play_pub, 'QT/happy', "Gesture")
            time.sleep(3)

            # Simulate user choice (50% chance to continue)
            print("Simulating user choice for second round...")
            time.sleep(3)
            if random.random() < 0.5:
                training_phase = training_phase_limit
                state.stop_metronome = False
                message = f'Great! Let us start {exercise_name} again!'
                safe_publish(speech_say_pub, message, "Speech")
                print("Simulated: User chose to continue")
            else:
                state.done = True
                message = 'I hope you enjoyed the exercise session with me. Thank you for your time and have a lovely day!'
                safe_publish(speech_say_pub, message, "Speech")
                safe_publish(gesture_play_pub, 'QT/happy', "Gesture")
                print("Simulated: User chose to end session")

        # End of session
        if not state.done and training_phase < 1:
            state.done = True
            message = f'We have reached the end of the {exercise_name} session. I hope you enjoyed exercising with me. Thank you!'
            safe_publish(speech_say_pub, message, "Speech")
            safe_publish(gesture_play_pub, 'QT/happy', "Gesture")

        time.sleep(1)

def read_data():
    """Read physiological data from CSV"""
    print("Starting data reader...")
    count = 0
    rmssd_prev = 100

    while not state.done and not rospy.is_shutdown():
        count += 1
        time.sleep(1)
        
        try:
            if os.path.exists("output.csv"):
                data = pd.read_csv("output.csv")
                
                if not data.empty and 'mean_hr' in data.columns:
                    state.mean_hr = data.mean_hr.iloc[-1]
                    
                    if count % 20 == 0 and 'rmssd' in data.columns:
                        current_rmssd = data.rmssd.iloc[-1]
                        rmssd_prev = current_rmssd
                else:
                    print("CSV file empty or missing columns")
            else:
                print("output.csv not found - creating dummy data")
                # Create dummy data for testing
                dummy_hr = 70 + random.uniform(-10, 20)
                state.mean_hr = dummy_hr
                
        except Exception as e:
            print(f"Data reading exception: {e}")

def main():
    """Main function to start the exercise system"""
    global speech_say_pub, emotion_show_pub, gesture_play_pub, audio_play_pub
    
    print("=== QT Robot Multi-Exercise Coach Starting ===")
    
    # Initialize ROS node
    try:
        rospy.init_node('qt_multi_exercise_coach', anonymous=True)
        print("✓ ROS node initialized")
        
        # Wait a moment for node to fully initialize
        time.sleep(1)
        
    except Exception as e:
        print(f"✗ Failed to initialize ROS node: {e}")
        print("Make sure roscore is running")
        return

    # Initialize ROS publishers
    try:
        speech_say_pub = rospy.Publisher('/qt_robot/speech/say', String, queue_size=10)
        emotion_show_pub = rospy.Publisher('/qt_robot/emotion/show', String, queue_size=10)
        gesture_play_pub = rospy.Publisher('/qt_robot/gesture/play', String, queue_size=10)
        audio_play_pub = rospy.Publisher('/qt_robot/audio/play', String, queue_size=10)
        
        print("✓ ROS publishers initialized")
        
        # Wait for publishers to connect
        time.sleep(2)
        
    except Exception as e:
        print(f"✗ Failed to initialize publishers: {e}")
        return

    # Start background threads
    pThread = Thread(target=read_data, daemon=True)
    cThread = Thread(target=calc_feedback, daemon=True)
    mThread = Thread(target=play_metronome, daemon=True)

    pThread.start()
    print("✓ Data reader started")

    # Initial greeting
    baseline_msg = ("Hello! Glad to see you here! Today, I will guide you through a personalized exercise session. "
                   "I have five different exercises for you to choose from: Squats, Lunges, High Knees, Push-Ups, and Burpees. "
                   "We will begin with collecting baseline data from your heart rate sensor, then a warm-up, "
                   "followed by your chosen exercise. Let's wait for 2 minutes to collect your baseline.")
    
    safe_publish(speech_say_pub, baseline_msg, "Speech")
    safe_publish(gesture_play_pub, 'QT/happy', "Gesture")
    
    # Play welcome audio
    safe_publish(audio_play_pub, "QT/Komiku_Glouglou", "Audio")

    # Start main threads
    cThread.start()
    mThread.start()
    print("✓ All systems started")

    try:
        # Main loop - use rospy.spin() or manual loop with rospy.is_shutdown() check
        while not state.done and not rospy.is_shutdown():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n=== Program interrupted by user ===")
        state.done = True

    # Cleanup
    print("=== Shutting down ===")
    state.done = True  # Signal threads to stop
    
    # Wait for threads to finish
    pThread.join(timeout=5)
    cThread.join(timeout=5)
    mThread.join(timeout=5)
    
    print("=== Multi-exercise session complete ===")

if __name__ == '__main__':
    main()
