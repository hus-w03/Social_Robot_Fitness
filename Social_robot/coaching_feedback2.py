#!/usr/bin/env python3
"""
Improved QT Robot Multi-Exercise Coach - Enhanced Speech Management
Heart Rate and HRV Monitoring System with Proper Speech Handling
Addresses speech interruption, timing, and personalized feedback issues
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

# Speech management settings
SPEECH_DELAY = 3.0  # Minimum delay between speech commands
GESTURE_DELAY = 1.5  # Delay after gestures
EMOTION_DELAY = 1.0  # Delay after emotions
FEEDBACK_COOLDOWN = 15.0  # Minimum time between exercise feedback

max_hr = 220 - AGE

# Exercise-specific configurations with enhanced feedback
EXERCISE_CONFIG = {
    "squats": {
        "name": "Squats",
        "instructions": "Stand with feet shoulder-width apart, lower your body as if sitting back into a chair, then return to standing.",
        "metronome_normal": 1.2,
        "metronome_fast": 0.8,
        "metronome_slow": 1.8,
        "intensity_modifier": 0.85,
        "duration": 60,
        "primary_emotion": "QT/happy",
        "encouraging_emotion": "QT/excited",
        "warning_emotion": "QT/sad",
        "primary_gesture": "QT/show_muscles",
        "encouraging_gesture": "QT/happy",
        "warning_gesture": "QT/attention",
        "motivational_phrases": [
            "Keep your chest up and core engaged!",
            "Remember to push through your heels!",
            "Great form! Keep those knees behind your toes!",
            "You're building strong legs!"
        ],
        "warning_phrases": [
            "Slow down on those squats",
            "Take it easy with the squats",
            "Reduce the squat intensity"
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
        "primary_emotion": "QT/happy",
        "encouraging_emotion": "QT/excited",
        "warning_emotion": "QT/confused",
        "primary_gesture": "QT/happy",
        "encouraging_gesture": "QT/show_muscles",
        "warning_gesture": "QT/attention",
        "motivational_phrases": [
            "Keep your torso upright!",
            "Feel the burn in those legs!",
            "Perfect form! Switch legs when ready!",
            "You're getting stronger with each rep!"
        ],
        "warning_phrases": [
            "Ease up on those lunges",
            "Slow down the lunge pace",
            "Take a breather from lunging"
        ]
    },
    "high_knees": {
        "name": "High Knees",
        "instructions": "Run in place, bringing your knees up toward your chest with each step.",
        "metronome_normal": 0.4,
        "metronome_fast": 0.25,
        "metronome_slow": 0.6,
        "intensity_modifier": 1.15,
        "duration": 60,
        "primary_emotion": "QT/excited",
        "encouraging_emotion": "QT/happy",
        "warning_emotion": "QT/sad",
        "primary_gesture": "QT/happy",
        "encouraging_gesture": "QT/excited",
        "warning_gesture": "QT/attention",
        "motivational_phrases": [
            "Get those knees up high!",
            "Pump those arms!",
            "Feel the cardio burn!",
            "You're flying! Keep it up!"
        ],
        "warning_phrases": [
            "Slow down those high knees",
            "Take it easier with the knees",
            "Reduce the knee pace"
        ]
    },
    "push_ups": {
        "name": "Push-Ups",
        "instructions": "Start in plank position, lower your body until chest nearly touches floor, then push back up.",
        "metronome_normal": 1.5,
        "metronome_fast": 1.0,
        "metronome_slow": 2.0,
        "intensity_modifier": 0.75,
        "duration": 60,
        "primary_emotion": "QT/happy",
        "encouraging_emotion": "QT/excited",
        "warning_emotion": "QT/confused",
        "primary_gesture": "QT/show_muscles",
        "encouraging_gesture": "QT/happy",
        "warning_gesture": "QT/attention",
        "motivational_phrases": [
            "Keep that core tight!",
            "Lower all the way down!",
            "Strong push! Feel those arms working!",
            "You're building upper body strength!"
        ],
        "warning_phrases": [
            "Take it easy on push-ups",
            "Slow down those push-ups",
            "Rest from the push-ups"
        ]
    },
    "burpees": {
        "name": "Burpees",
        "instructions": "Start standing, drop to squat, jump back to plank, do push-up, jump feet to squat, then jump up with arms overhead.",
        "metronome_normal": 2.0,
        "metronome_fast": 1.5,
        "metronome_slow": 3.0,
        "intensity_modifier": 1.25,
        "duration": 45,
        "primary_emotion": "QT/excited",
        "encouraging_emotion": "QT/happy",
        "warning_emotion": "QT/sad",
        "primary_gesture": "QT/excited",
        "encouraging_gesture": "QT/show_muscles",
        "warning_gesture": "QT/attention",
        "motivational_phrases": [
            "Full body power! You've got this!",
            "Each burpee makes you stronger!",
            "Feel that total body burn!",
            "You're a warrior! Keep fighting!"
        ],
        "warning_phrases": [
            "Slow down those burpees",
            "Take it easier with burpees",
            "Rest from the burpees"
        ]
    }
}

# Thread-safe state management with speech control
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
        
        # Speech management
        self.last_speech_time = 0
        self.last_feedback_time = 0
        self.speech_active = False
    
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

def safe_publish(publisher, message_data, topic_name="Unknown", wait_time=0):
    """Safely publish to ROS topic with error handling and timing control"""
    try:
        if publisher is not None:
            msg = String()
            msg.data = message_data
            publisher.publish(msg)
            print(f"[{topic_name}] {message_data}")
            if wait_time > 0:
                time.sleep(wait_time)
            return True
        else:
            print(f"[ERROR] {topic_name} publisher not initialized")
            return False
    except Exception as e:
        print(f"[ERROR] Failed to publish to {topic_name}: {e}")
        return False

def robot_speak_managed(message, priority="normal"):
    """
    Managed speech function with interruption handling and timing control
    
    Args:
        message (str): Text to speak
        priority (str): "high", "normal", or "low" - high priority can interrupt
    """
    current_time = time.time()
    
    # Check if enough time has passed since last speech
    if priority != "high" and (current_time - state.last_speech_time) < SPEECH_DELAY:
        print(f"[SPEECH SKIPPED] Too soon since last speech: {message}")
        return False
    
    # High priority can interrupt current speech
    if priority == "high":
        # Send speech stop command (if your robot supports it)
        # safe_publish(speech_say_pub, "", "Speech Stop")
        time.sleep(0.5)  # Brief pause for interruption
    
    # Send the speech command
    success = safe_publish(speech_say_pub, message, "Speech", SPEECH_DELAY)
    
    if success:
        state.last_speech_time = current_time
        state.speech_active = True
        
        # Calculate speech duration (rough estimate: 150 words per minute)
        words = len(message.split())
        estimated_duration = (words / 150.0) * 60.0
        
        # Schedule speech completion
        def mark_speech_complete():
            time.sleep(estimated_duration)
            state.speech_active = False
        
        speech_thread = Thread(target=mark_speech_complete, daemon=True)
        speech_thread.start()
    
    return success

def robot_show_emotion(emotion, exercise_context=False):
    """Show emotion with context awareness"""
    if exercise_context and state.current_exercise:
        config = EXERCISE_CONFIG[state.current_exercise]
        emotion = config.get("primary_emotion", emotion)
    
    safe_publish(emotion_show_pub, emotion, "Emotion", EMOTION_DELAY)

def robot_play_gesture(gesture, exercise_context=False):
    """Play gesture with context awareness"""
    if exercise_context and state.current_exercise:
        config = EXERCISE_CONFIG[state.current_exercise]
        gesture = config.get("primary_gesture", gesture)
    
    safe_publish(gesture_play_pub, gesture, "Gesture", GESTURE_DELAY)

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
    """Handle exercise selection process with proper timing"""
    print("Starting exercise selection...")
    
    exercises = list(EXERCISE_CONFIG.keys())
    
    robot_speak_managed("Great! Now let's choose your exercise. I have five options for you.", "high")
    time.sleep(2)  # Pause for user to process
    
    for i, exercise_key in enumerate(exercises, 1):
        exercise_name = EXERCISE_CONFIG[exercise_key]["name"]
        robot_speak_managed(f"Option {i}: {exercise_name}", "normal")
        time.sleep(1.5)  # Pause between options
    
    robot_speak_managed("Please tell me which number you'd like to choose, from 1 to 5.", "normal")
    
    # Extended pause for user decision
    print("Waiting for user selection...")
    time.sleep(8)
    
    # Simulate user selection (replace with actual input mechanism)
    selected_index = random.randint(0, len(exercises) - 1)
    selected_exercise = exercises[selected_index]
    
    exercise_name = EXERCISE_CONFIG[selected_exercise]["name"]
    instructions = EXERCISE_CONFIG[selected_exercise]["instructions"]
    
    robot_speak_managed(f"Excellent choice! You selected {exercise_name}.", "normal")
    time.sleep(2)
    
    robot_speak_managed(f"Here's how to do {exercise_name}: {instructions}", "normal")
    robot_show_emotion("QT/happy", exercise_context=True)
    robot_play_gesture("QT/happy", exercise_context=True)
    
    state.current_exercise = selected_exercise
    state.exercise_selected = True
    
    print(f"Selected exercise: {exercise_name}")
    time.sleep(3)  # Allow time for user to understand instructions
    return selected_exercise

def speak_feedback(warm_up, give_feedback, warm_up_count):
    """Provide feedback with enhanced speech management and personalization"""
    current_rest_hr = state.rest_hr
    current_mean_hr = state.mean_hr
    current_exercise = state.current_exercise
    current_time = time.time()
    
    # Check if we have valid data
    if current_rest_hr is None or current_mean_hr == -1:
        print("Waiting for valid heart rate data...")
        return 0

    # Check feedback cooldown
    if not warm_up and (current_time - state.last_feedback_time) < FEEDBACK_COOLDOWN:
        return 0

    # Get exercise-specific thresholds and config
    thresholds = get_exercise_thresholds(current_exercise)
    if not thresholds:
        return 0

    min_thresh = thresholds["min_thresh"]
    max_thresh = thresholds["max_thresh"]
    upper_third = thresholds["upper_third"]
    lower_third = thresholds["lower_third"]

    # Exercise-specific phrases and feedback
    if current_exercise and current_exercise in EXERCISE_CONFIG:
        config = EXERCISE_CONFIG[current_exercise]
        exercise_phrases = config["motivational_phrases"]
        warning_phrases = config["warning_phrases"]
        encouraging_emotion = config["encouraging_emotion"]
        warning_emotion = config["warning_emotion"]
        encouraging_gesture = config["encouraging_gesture"]
        warning_gesture = config["warning_gesture"]
    else:
        exercise_phrases = ["Keep going!", "You're doing great!", "Stay strong!"]
        warning_phrases = ["Slow down!", "Take it easy!", "Rest a bit!"]
        encouraging_emotion = "QT/happy"
        warning_emotion = "QT/sad"
        encouraging_gesture = "QT/happy"
        warning_gesture = "QT/attention"

    print(f"Feedback - Exercise: {current_exercise}, Warm-up: {warm_up}, HR: {current_mean_hr:.1f}")
    
    if warm_up:
        print(f"Warm-up count: {warm_up_count}")

        if current_mean_hr <= current_rest_hr:
            if warm_up_count < 1:
                robot_speak_managed("Let's increase the pace a bit!", "normal")
                robot_show_emotion(encouraging_emotion)
                robot_play_gesture(encouraging_gesture)
            return 0

        if current_rest_hr < current_mean_hr < max_thresh:
            if warm_up_count < 1:
                robot_speak_managed(random.choice(exercise_phrases), "normal")
                robot_show_emotion(encouraging_emotion)
                robot_play_gesture(encouraging_gesture)
            return 0

        if current_mean_hr > max_thresh:
            if warm_up_count < 1:
                robot_speak_managed('Looks like you are ready to start the exercise!', "normal")
                robot_show_emotion("QT/excited")
                robot_play_gesture("QT/excited")
            return 0

    if not warm_up:
        if current_mean_hr > max_thresh:
            if give_feedback == 10:
                robot_speak_managed(random.choice(warning_phrases), "high")  # High priority for safety
                robot_show_emotion(warning_emotion)
                robot_play_gesture(warning_gesture)
                state.last_feedback_time = current_time
            return 0

        if current_mean_hr < min_thresh:
            if give_feedback < 1:
                exercise_name = EXERCISE_CONFIG[current_exercise]["name"]
                robot_speak_managed(f'You are resting, lets continue with {exercise_name}!', "normal")
                robot_show_emotion(encouraging_emotion)
                robot_play_gesture(encouraging_gesture)
                state.last_feedback_time = current_time
            return 0

        if current_mean_hr > upper_third:
            if give_feedback < 1:
                if not state.inc_speed:
                    robot_speak_managed('You are doing great! Would you like to slow down?', "normal")
                    robot_show_emotion(encouraging_emotion)
                    robot_play_gesture(encouraging_gesture)
                else:
                    robot_speak_managed(random.choice(exercise_phrases), "normal")
                    robot_show_emotion(encouraging_emotion)
                    robot_play_gesture(encouraging_gesture)

                # Extended pause for user response
                time.sleep(5)
                # Simulate user response (30% chance to slow down)
                if random.random() < 0.3:
                    state.slow_down = True
                    print("Simulated: User chose to slow down")
                else:
                    print("Simulated: User chose not to slow down")
                state.last_feedback_time = current_time
            return 0

        if current_mean_hr > min_thresh or current_mean_hr > lower_third:
            if give_feedback < 1:
                if not state.inc_speed:
                    robot_speak_managed("You are doing good! Would you like to go faster?", "normal")
                    robot_show_emotion(encouraging_emotion)
                    robot_play_gesture(encouraging_gesture)
                else:
                    robot_speak_managed(random.choice(exercise_phrases), "normal")
                    robot_show_emotion(encouraging_emotion)
                    robot_play_gesture(encouraging_gesture)

                # Extended pause for user response
                time.sleep(5)
                # Simulate user response (20% chance to increase speed)
                if random.random() < 0.2:
                    state.inc_speed = True
                    print("Simulated: User chose to increase speed")
                else:
                    print("Simulated: User chose not to increase speed")
                state.last_feedback_time = current_time
            return 0

def calc_initial_hr():
    """Calculate baseline heart rate with better speech pacing"""
    print("Calculating initial heart rate...")
    
    robot_speak_managed("Now I will collect your baseline heart rate. Please remain calm and relaxed.", "high")
    time.sleep(3)
    
    while state.mean_hr == -1:
        print("Waiting for heart rate data...")
        robot_speak_managed("Please wait while I read your heart rate sensor.", "normal")
        time.sleep(15)  # Longer wait for sensor data

    state.rest_hr = state.mean_hr
    print(f"Resting HR: {state.rest_hr:.1f}")
    print(f"Max HR: {max_hr}")
    print(f"Upper HR limit: {(MAX_THRESH * (max_hr - state.rest_hr)) + state.rest_hr:.1f}")
    print(f"Lower HR limit: {(MIN_THRESH * (max_hr - state.rest_hr)) + state.rest_hr:.1f}")
    
    robot_speak_managed(f"Perfect! Your resting heart rate is {state.rest_hr:.0f} beats per minute.", "normal")
    robot_show_emotion("QT/happy")
    robot_play_gesture("QT/happy")
    time.sleep(2)

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
    """Main exercise feedback logic with improved timing"""
    print("Starting exercise feedback system...")
    
    calc_initial_hr()
    
    robot_speak_managed("The baseline collection is complete. Before you start, make sure you have enough space around you.", "normal")
    time.sleep(8)  # Extended pause for user preparation

    # Exercise selection phase
    selected_exercise = select_exercise()
    exercise_config = EXERCISE_CONFIG[selected_exercise]
    
    time.sleep(5)  # Pause before starting warmup

    # Exercise parameters
    training_phase_limit = exercise_config["duration"]
    give_feedback_limit = 12
    warm_up_time = 30
    warm_up_limit = 20
    give_feedback_count = give_feedback_limit
    warm_up_count = warm_up_limit

    # Start warm-up
    robot_speak_managed('Lets start a warm-up! You can start with light movement to prepare for your exercise.', "high")
    robot_show_emotion("QT/happy", exercise_context=True)
    robot_play_gesture("QT/happy", exercise_context=True)
    time.sleep(3)

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
            
        time.sleep(2)  # Slower execution for warmup

    # Main exercise phase
    training_phase = training_phase_limit
    state.first_time = True
    time.sleep(3)

    exercise_name = exercise_config["name"]
    robot_speak_managed(f'Let us start {exercise_name}! I am playing a metronome to help with pacing.', "high")
    robot_show_emotion("QT/excited", exercise_context=True)
    robot_play_gesture("QT/excited", exercise_context=True)
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
            
            robot_speak_managed(f'Great work on {exercise_name}! Lets take thirty seconds to catch our breath!', "high")
            robot_show_emotion("QT/happy", exercise_context=True)
            robot_play_gesture("QT/happy", exercise_context=True)
            time.sleep(35)  # Extended rest period

            robot_speak_managed('If you want to end this session, let me know! Otherwise, we can go for a second round!', "normal")
            robot_show_emotion("QT/happy", exercise_context=True)
            robot_play_gesture("QT/happy", exercise_context=True)
            time.sleep(8)  # Extended pause for user decision

            # Simulate user choice (50% chance to continue)
            print("Simulating user choice for second round...")
            time.sleep(5)
            if random.random() < 0.5:
                training_phase = training_phase_limit
                state.stop_metronome = False
                robot_speak_managed(f'Great! Let us start {exercise_name} again!', "normal")
                robot_show_emotion("QT/excited", exercise_context=True)
                robot_play_gesture("QT/excited", exercise_context=True)
                print("Simulated: User chose to continue")
            else:
                state.done = True
                robot_speak_managed('I hope you enjoyed the exercise session with me. Thank you for your time and have a lovely day!', "normal")
                robot_show_emotion("QT/happy")
                robot_play_gesture("QT/happy")
                print("Simulated: User chose to end session")

        # End of session
        if not state.done and training_phase < 1:
            state.done = True
            robot_speak_managed(f'We have reached the end of the {exercise_name} session. I hope you enjoyed exercising with me. Thank you!', "normal")
            robot_show_emotion("QT/happy", exercise_context=True)
            robot_play_gesture("QT/happy", exercise_context=True)

        time.sleep(2)  # Slower main loop execution

def read_data():
    """Read physiological data from CSV"""
    print("Starting data reader...")
    count = 0
    rmssd_prev = 100

    while not state.done and not rospy.is_shutdown():
        count += 1
        time.sleep(2)  # Slower data reading
        
        try:
            if os.path.exists("output.csv"):
                data = pd.read_csv("output.csv")
                
                if not data.empty and 'mean_hr' in data.columns:
                    state.mean_hr = data.mean_hr.iloc[-1]
                    
                    if count % 30 == 0 and 'rmssd' in data.columns:  # Less frequent HRV updates
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
    
    print("=== QT Robot Multi-Exercise Coach with Enhanced Speech Management ===")
    
    # Initialize ROS node
    try:
        rospy.init_node('qt_enhanced_exercise_coach', anonymous=True)
        print("✓ ROS node initialized")
        time.sleep(2)  # Extended initialization pause
        
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
        time.sleep(3)  # Extended publisher connection time
        
    except Exception as e:
        print(f"✗ Failed to initialize publishers: {e}")
        return

    # Start background threads
    pThread = Thread(target=read_data, daemon=True)
    cThread = Thread(target=calc_feedback, daemon=True)
    mThread = Thread(target=play_metronome, daemon=True)

    pThread.start()
    print("✓ Data reader started")

    # Initial greeting with enhanced speech management
    baseline_msg = ("Hello! Glad to see you here! Today, I will guide you through a personalized exercise session. "
                   "I have five different exercises for you to choose from. "
                   "We will begin with collecting baseline data from your heart rate sensor.")
    
    robot_speak_managed(baseline_msg, "high")
    robot_show_emotion("QT/happy")
    robot_play_gesture("QT/happy")
    
    # Play welcome audio
    safe_publish(audio_play_pub, "QT/Komiku_Glouglou", "Audio")

    # Start main threads
    cThread.start()
    mThread.start()
    print("✓ All systems started with enhanced speech management")

    try:
        while not state.done and not rospy.is_shutdown():
            time.sleep(2)  # Slower main loop
    except KeyboardInterrupt:
        print("\n=== Program interrupted by user ===")
        state.done = True

    # Cleanup
    print("=== Shutting down ===")
    state.done = True
    
    # Wait for threads to finish
    pThread.join(timeout=10)
    cThread.join(timeout=10)
    mThread.join(timeout=10)
    
    print("=== Enhanced multi-exercise session complete ===")

if __name__ == '__main__':
    main()