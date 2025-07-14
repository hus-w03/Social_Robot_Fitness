#!/usr/bin/env python3
"""
Terminal-Based Multi-Exercise Coach with HRV Integration
Heart Rate and HRV (RMSSD) Monitoring System for Terminal Testing
Implements HRV Safety Override system with exercise-specific thresholds
"""

import time
import random
from threading import Thread, Lock
import pandas as pd
import os
import sys
from datetime import datetime
import numpy as np

# User configuration
AGE = 35
MAX_THRESH = .80
UPPER_THIRD = 0.75
LOWER_THIRD = 0.65
MIN_THRESH = .40

# HRV (RMSSD) Safety Thresholds (percentage of baseline)
HRV_GREEN_ZONE = 0.60    # >60% of baseline - good autonomic function
HRV_YELLOW_ZONE = 0.30   # 30-60% of baseline - increasing stress
HRV_RED_ZONE = 0.20      # 20-30% of baseline - high stress
HRV_CRITICAL = 0.15      # <15% of baseline - critical, stop exercise

max_hr = 220 - AGE

# Exercise-specific configurations with HRV modifiers
EXERCISE_CONFIG = {
    "squats": {
        "name": "Squats",
        "instructions": "Stand with feet shoulder-width apart, lower your body as if sitting back into a chair, then return to standing.",
        "metronome_normal": 1.2,
        "metronome_fast": 0.8,
        "metronome_slow": 1.8,
        "intensity_modifier": 0.85,
        "hrv_modifier": 1.1,  # Allow slightly higher HRV threshold (less autonomic stress)
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
        "hrv_modifier": 1.05,  # Slightly higher HRV threshold
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
        "metronome_normal": 0.4,
        "metronome_fast": 0.25,
        "metronome_slow": 0.6,
        "intensity_modifier": 1.15,
        "hrv_modifier": 0.85,  # Lower HRV threshold (expect more autonomic stress)
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
        "metronome_normal": 1.5,
        "metronome_fast": 1.0,
        "metronome_slow": 2.0,
        "intensity_modifier": 0.75,
        "hrv_modifier": 1.2,  # Highest HRV threshold (mainly strength, less cardio stress)
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
        "metronome_normal": 2.0,
        "metronome_fast": 1.5,
        "metronome_slow": 3.0,
        "intensity_modifier": 1.25,
        "hrv_modifier": 0.75,  # Lowest HRV threshold (highest autonomic stress expected)
        "duration": 45,
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
        self._rest_hrv = None  # Baseline HRV (RMSSD)
        self._current_hrv = -1  # Current HRV reading
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
        self.user_input_requested = False
        self.hrv_warning_active = False
    
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
    def rest_hrv(self):
        with self.lock:
            return self._rest_hrv
    
    @rest_hrv.setter
    def rest_hrv(self, value):
        with self.lock:
            self._rest_hrv = value
    
    @property
    def current_hrv(self):
        with self.lock:
            return self._current_hrv
    
    @current_hrv.setter
    def current_hrv(self, value):
        with self.lock:
            self._current_hrv = value
    
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

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def print_coach_message(message):
    """Print coach message with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] 🤖 COACH: {message}")

def print_status(message):
    """Print status message"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] ℹ️  STATUS: {message}")

def print_hr_data(hr, hrv=None, exercise=None):
    """Print heart rate and HRV data with context"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    if hrv is not None and hrv > 0:
        if exercise:
            print(f"[{timestamp}] ❤️  HR: {hr:.1f} BPM | HRV: {hrv:.1f} ms ({exercise})")
        else:
            print(f"[{timestamp}] ❤️  HR: {hr:.1f} BPM | HRV: {hrv:.1f} ms")
    else:
        if exercise:
            print(f"[{timestamp}] ❤️  HR: {hr:.1f} BPM ({exercise})")
        else:
            print(f"[{timestamp}] ❤️  HR: {hr:.1f} BPM")

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

def get_hrv_thresholds(exercise_name):
    """Calculate exercise-specific HRV thresholds"""
    if state.rest_hrv is None:
        return None
    
    config = EXERCISE_CONFIG.get(exercise_name, EXERCISE_CONFIG["squats"])
    hrv_modifier = config["hrv_modifier"]
    
    # Apply exercise-specific modifier to base thresholds
    return {
        "green_zone": state.rest_hrv * HRV_GREEN_ZONE * hrv_modifier,
        "yellow_zone": state.rest_hrv * HRV_YELLOW_ZONE * hrv_modifier,
        "red_zone": state.rest_hrv * HRV_RED_ZONE * hrv_modifier,
        "critical": state.rest_hrv * HRV_CRITICAL * hrv_modifier
    }

def analyze_hrv_status(current_hrv, exercise_name):
    """Analyze current HRV status and return warning level"""
    if current_hrv <= 0 or state.rest_hrv is None:
        return "unknown", "No HRV data available"
    
    hrv_thresholds = get_hrv_thresholds(exercise_name)
    if not hrv_thresholds:
        return "unknown", "Unable to calculate HRV thresholds"
    
    hrv_percentage = (current_hrv / state.rest_hrv) * 100
    
    if current_hrv >= hrv_thresholds["green_zone"]:
        return "green", f"HRV normal ({hrv_percentage:.1f}% of baseline)"
    elif current_hrv >= hrv_thresholds["yellow_zone"]:
        return "yellow", f"HRV decreasing ({hrv_percentage:.1f}% of baseline) - monitor closely"
    elif current_hrv >= hrv_thresholds["red_zone"]:
        return "red", f"HRV low ({hrv_percentage:.1f}% of baseline) - consider slowing down"
    else:
        return "critical", f"HRV critically low ({hrv_percentage:.1f}% of baseline) - stop exercise!"

def get_user_input(prompt, valid_options=None):
    """Get user input with validation"""
    while True:
        try:
            response = input(f"\n🎤 {prompt}: ").strip().lower()
            
            if valid_options:
                if response in valid_options:
                    return response
                else:
                    print(f"❌ Please enter one of: {', '.join(valid_options)}")
            else:
                return response
                
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted by user. Goodbye!")
            state.done = True
            sys.exit(0)

def select_exercise():
    """Handle exercise selection process"""
    print_header("EXERCISE SELECTION")
    
    exercises = list(EXERCISE_CONFIG.keys())
    
    print_coach_message("Great! Now let's choose your exercise. I have five options for you:")
    
    for i, exercise_key in enumerate(exercises, 1):
        exercise_name = EXERCISE_CONFIG[exercise_key]["name"]
        instructions = EXERCISE_CONFIG[exercise_key]["instructions"]
        hrv_modifier = EXERCISE_CONFIG[exercise_key]["hrv_modifier"]
        print(f"\n{i}. {exercise_name}")
        print(f"   📝 {instructions}")
        print(f"   🧠 HRV Sensitivity: {'High' if hrv_modifier > 1.0 else 'Medium' if hrv_modifier > 0.9 else 'Low'}")
    
    while True:
        try:
            choice = get_user_input("Please enter the number of your chosen exercise (1-5)")
            choice_num = int(choice)
            
            if 1 <= choice_num <= 5:
                selected_exercise = exercises[choice_num - 1]
                exercise_name = EXERCISE_CONFIG[selected_exercise]["name"]
                instructions = EXERCISE_CONFIG[selected_exercise]["instructions"]
                
                print_coach_message(f"Excellent choice! You selected {exercise_name}.")
                print_coach_message(f"Remember: {instructions}")
                
                state.current_exercise = selected_exercise
                state.exercise_selected = True
                
                print_status(f"Selected exercise: {exercise_name}")
                return selected_exercise
            else:
                print("❌ Please enter a number between 1 and 5.")
                
        except ValueError:
            print("❌ Please enter a valid number.")

def speak_feedback(warm_up, give_feedback, warm_up_count):
    """Provide feedback based on heart rate, HRV, and current exercise"""
    current_rest_hr = state.rest_hr
    current_mean_hr = state.mean_hr
    current_hrv = state.current_hrv
    current_exercise = state.current_exercise
    
    # Check if we have valid data
    if current_rest_hr is None or current_mean_hr == -1:
        print_status("Waiting for valid heart rate data...")
        return 0

    # HRV Safety Override Check (PRIORITY #1)
    if current_hrv > 0 and current_exercise:
        hrv_status, hrv_message = analyze_hrv_status(current_hrv, current_exercise)
        
        if hrv_status == "critical":
            print_coach_message(f"🚨 CRITICAL HRV WARNING: {hrv_message}")
            print_coach_message("Please stop the exercise immediately and rest!")
            state.hrv_warning_active = True
            return 0
        elif hrv_status == "red":
            if give_feedback == 10 or not state.hrv_warning_active:
                print_coach_message(f"⚠️  HRV ALERT: {hrv_message}")
                print_coach_message("Your body is showing high stress. Consider slowing down.")
                state.hrv_warning_active = True
            return 0
        elif hrv_status == "yellow":
            if give_feedback < 1:
                print_coach_message(f"🟡 HRV NOTICE: {hrv_message}")
                state.hrv_warning_active = False
        else:  # green or unknown
            state.hrv_warning_active = False

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

    # Add HRV context to feedback when appropriate
    hrv_context = ""
    if current_hrv > 0 and state.rest_hrv:
        hrv_percentage = (current_hrv / state.rest_hrv) * 100
        if hrv_percentage > 70:
            hrv_context = " Your stress levels look good!"
        elif hrv_percentage > 50:
            hrv_context = " Your body is handling this well."

    if warm_up:
        if current_mean_hr <= current_rest_hr:
            if warm_up_count < 1:
                print_coach_message(random.choice(inc_statements) + hrv_context)
            return 0

        if current_rest_hr < current_mean_hr < max_thresh:
            if warm_up_count < 1:
                print_coach_message(random.choice(exercise_phrases) + hrv_context)
            return 0

        if current_mean_hr > max_thresh:
            if warm_up_count < 1:
                print_coach_message('Looks like you are ready to start the exercise!' + hrv_context)
            return 0

    if not warm_up:
        # Skip other feedback if HRV warning is active
        if state.hrv_warning_active:
            return 0
            
        if current_mean_hr > max_thresh:
            if give_feedback == 10:
                print_coach_message('⚠️  Heart rate is high! Slow down and dont over exert yourself.')
            return 0

        if current_mean_hr < min_thresh:
            if give_feedback < 1:
                exercise_name = EXERCISE_CONFIG[current_exercise]["name"]
                print_coach_message(f'You are resting, lets continue with {exercise_name}!' + hrv_context)
            return 0

        if current_mean_hr > upper_third:
            if give_feedback < 1:
                if not state.inc_speed:
                    message = 'You are doing great, would you like to slow down?'
                    if hrv_context:
                        message += hrv_context
                    print_coach_message(message)
                    response = get_user_input("Type 'yes' to slow down, or 'no' to continue", ['yes', 'no'])
                    if response == 'yes':
                        state.slow_down = True
                        print_coach_message("Okay, slowing down the pace!")
                    else:
                        print_coach_message("Great! Keep up the intensity!")
                else:
                    print_coach_message(random.choice(exercise_phrases) + hrv_context)
            return 0

        if current_mean_hr > min_thresh or current_mean_hr > lower_third:
            if give_feedback < 1:
                if not state.inc_speed:
                    message = random.choice(mot_statements)
                    if hrv_context:
                        message += hrv_context
                    print_coach_message(message)
                    response = get_user_input("Type 'yes' to increase pace, or 'no' to continue", ['yes', 'no'])
                    if response == 'yes':
                        state.inc_speed = True
                        print_coach_message("Let's push harder! Increasing the pace!")
                    else:
                        print_coach_message("Perfect! Maintain your current pace!")
                else:
                    print_coach_message(random.choice(exercise_phrases) + hrv_context)
            return 0

def calc_initial_hr():
    """Calculate baseline heart rate and HRV"""
    print_header("BASELINE COLLECTION - HR & HRV")
    print_coach_message("Calculating your baseline heart rate and heart rate variability...")
    print_status("Please remain calm and relaxed while we collect baseline data...")
    print_status("📊 This will take 60 seconds for accurate HRV measurement")
    
    baseline_duration = 60  # Longer duration for accurate HRV baseline
    start_time = time.time()
    
    hr_readings = []
    hrv_readings = []
    
    while time.time() - start_time < baseline_duration:
        if state.mean_hr == -1:
            print_status("Waiting for heart rate data...")
            time.sleep(5)
        else:
            remaining = baseline_duration - (time.time() - start_time)
            hr_readings.append(state.mean_hr)
            
            if state.current_hrv > 0:
                hrv_readings.append(state.current_hrv)
                print_status(f"Collecting baseline... {remaining:.0f}s remaining (HR: {state.mean_hr:.1f}, HRV: {state.current_hrv:.1f})")
            else:
                print_status(f"Collecting baseline... {remaining:.0f}s remaining (HR: {state.mean_hr:.1f}, HRV: waiting...)")
            time.sleep(2)

    # Calculate baseline values
    if hr_readings:
        state.rest_hr = np.mean(hr_readings)
        print_coach_message(f"✅ Baseline heart rate: {state.rest_hr:.1f} BPM")
    else:
        print_coach_message("⚠️  Unable to collect baseline HR. Using default values.")
        state.rest_hr = 70
    
    if hrv_readings:
        state.rest_hrv = np.mean(hrv_readings)
        print_coach_message(f"✅ Baseline HRV (RMSSD): {state.rest_hrv:.1f} ms")
    else:
        print_coach_message("⚠️  Unable to collect baseline HRV. HRV monitoring will be limited.")
        state.rest_hrv = 30  # Default baseline HRV
    
    state.baseline_complete = True
    
    print(f"📊 Your baseline metrics:")
    print(f"   • Resting HR: {state.rest_hr:.1f} BPM")
    print(f"   • Max HR: {max_hr} BPM")
    if state.rest_hrv:
        print(f"   • Baseline HRV: {state.rest_hrv:.1f} ms")
        print(f"   • HRV Green Zone: >{state.rest_hrv * HRV_GREEN_ZONE:.1f} ms")
        print(f"   • HRV Yellow Zone: {state.rest_hrv * HRV_YELLOW_ZONE:.1f}-{state.rest_hrv * HRV_GREEN_ZONE:.1f} ms")
        print(f"   • HRV Red Zone: <{state.rest_hrv * HRV_YELLOW_ZONE:.1f} ms")
        
    # Show exercise-specific thresholds for current exercise
    if state.current_exercise:
        hr_thresholds = get_exercise_thresholds(state.current_exercise)
        hrv_thresholds = get_hrv_thresholds(state.current_exercise)
        if hr_thresholds and hrv_thresholds:
            exercise_name = EXERCISE_CONFIG[state.current_exercise]['name']
            print(f"📊 {exercise_name} thresholds:")
            print(f"   • HR Target: {hr_thresholds['lower_third']:.1f}-{hr_thresholds['upper_third']:.1f} BPM")
            print(f"   • HR Max: {hr_thresholds['max_thresh']:.1f} BPM")
            print(f"   • HRV Warning: <{hrv_thresholds['yellow_zone']:.1f} ms")
            print(f"   • HRV Critical: <{hrv_thresholds['critical']:.1f} ms")

def virtual_metronome():
    """Visual metronome in terminal"""
    print_header("METRONOME STARTED")
    print_status("Following the rhythm: 🔴 = Beat")
    
    while not state.done:
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
                print("🔴 (SLOW)", end=" ", flush=True)
                time.sleep(slow_tempo)
            elif state.inc_speed:
                print("🔴 (FAST)", end=" ", flush=True)
                time.sleep(fast_tempo)
            elif state.stop_metronome:
                print("⚪ (REST)", end=" ", flush=True)
                time.sleep(0.5)
            elif not state.warm_up_start and state.exercise_selected:
                print("🔴", end=" ", flush=True)
                time.sleep(normal_tempo)
            else:
                time.sleep(0.5)
                
        except Exception as e:
            print_status(f"Metronome error: {e}")
            time.sleep(0.5)

def calc_feedback():
    """Main exercise feedback logic"""
    print_header("EXERCISE FEEDBACK SYSTEM")
    
    calc_initial_hr()
    
    print_coach_message("The baseline collection is complete. Before you start, make sure you have enough space around you.")
    input("\n⏸️  Press Enter when you're ready to continue...")

    # Exercise selection phase
    selected_exercise = select_exercise()
    exercise_config = EXERCISE_CONFIG[selected_exercise]
    
    input("\n⏸️  Press Enter when you're ready to start the warm-up...")

    # Exercise parameters
    training_phase_limit = exercise_config["duration"]
    give_feedback_limit = 12
    warm_up_time = 30
    warm_up_limit = 20
    give_feedback_count = give_feedback_limit
    warm_up_count = warm_up_limit

    # Start warm-up
    print_header("WARM-UP PHASE")
    print_coach_message('Lets start a warm-up! You can start with light movement to prepare for your exercise.')
    print_status("🧠 Monitoring both heart rate and HRV during warm-up")
    
    warm_up_start_time = time.time()

    # Warm-up phase
    while state.warm_up_start:
        warm_up_count -= 1
        warm_up_time -= 1
        elapsed_warmup = time.time() - warm_up_start_time
        
        print_status(f"Warm-up in progress... {warm_up_time}s remaining")
        
        if state.rest_hr is not None and state.mean_hr > 0:
            hr_thresholds = get_exercise_thresholds(selected_exercise)
            if hr_thresholds:
                print_hr_data(state.mean_hr, state.current_hrv, "WARM-UP")

        speak_feedback(state.warm_up_start, 0, warm_up_count)
        
        if warm_up_count < 1:
            warm_up_count = warm_up_limit
        if warm_up_time < 1:
            state.warm_up_start = False
            
        time.sleep(1)

    # Main exercise phase
    print_header(f"MAIN EXERCISE: {exercise_config['name'].upper()}")
    training_phase = training_phase_limit
    state.first_time = True

    exercise_name = exercise_config["name"]
    print_coach_message(f'Let us start {exercise_name}! Follow the metronome rhythm below for pacing.')
    print_status("🧠 Real-time HR and HRV monitoring active - I'll warn you if stress levels get too high")
    
    # Start virtual metronome in background
    metronome_thread = Thread(target=virtual_metronome, daemon=True)
    metronome_thread.start()
    
    input("\n⏸️  Press Enter to begin the exercise...")
    exercise_start_time = time.time()

    # Exercise loop
    while not state.done:
        training_phase -= 1
        give_feedback_count -= 1
        elapsed_exercise = time.time() - exercise_start_time

        if training_phase > 1:
            speak_feedback(False, give_feedback_count, 100)
            
        if state.rest_hr is not None:
            hrv_display = f", HRV: {state.current_hrv:.1f}" if state.current_hrv > 0 else ""
            print_status(f"Exercise: {training_phase}s remaining, HR: {state.mean_hr:.1f}{hrv_display}")

        if give_feedback_count < 1:
            give_feedback_count = give_feedback_limit

        # Rest break after first round
        if training_phase == 0 and state.first_time:
            state.first_time = False
            state.inc_speed = False
            state.slow_down = False
            state.stop_metronome = True
            state.hrv_warning_active = False
            
            print_header("REST BREAK")
            print_coach_message(f'Great work on {exercise_name}! Lets take thirty seconds to catch our breath!')
            print_status("🧠 Monitoring HRV recovery during rest...")
            
            # 30 second rest with countdown and HRV monitoring
            for i in range(30, 0, -1):
                hrv_recovery = ""
                if state.current_hrv > 0 and state.rest_hrv:
                    recovery_percent = (state.current_hrv / state.rest_hrv) * 100
                    hrv_recovery = f" | HRV Recovery: {recovery_percent:.1f}%"
                print(f"\r⏱️  Rest time: {i}s remaining{hrv_recovery}", end="", flush=True)
                time.sleep(1)
            print()

            continue_choice = get_user_input("Would you like to do a second round?", ['yes', 'no'])
            
            if continue_choice == 'yes':
                training_phase = training_phase_limit
                state.stop_metronome = False
                print_coach_message(f'Great! Let us start {exercise_name} again!')
            else:
                state.done = True
                print_coach_message('I hope you enjoyed the exercise session with me. Thank you for your time and have a lovely day!')

        # End of session
        if not state.done and training_phase < 1:
            state.done = True
            print_header("SESSION COMPLETE")
            print_coach_message(f'We have reached the end of the {exercise_name} session. I hope you enjoyed exercising with me. Thank you!')

        time.sleep(1)

def read_data():
    """Read physiological data from CSV including HRV"""
    print_status("Starting data reader...")
    count = 0
    last_hr = -1
    last_hrv = -1

    while not state.done:
        count += 1
        time.sleep(1)
        
        try:
            if os.path.exists("output.csv"):
                data = pd.read_csv("output.csv")
                
                if not data.empty:
                    # Read heart rate
                    if 'mean_hr' in data.columns:
                        new_hr = data.mean_hr.iloc[-1]
                        if new_hr != last_hr:
                            state.mean_hr = new_hr
                            last_hr = new_hr
                    
                    # Read HRV 
                    hrv_columns = ['rmssd']
                    for col in hrv_columns:
                        if col in data.columns:
                            new_hrv = data[col].iloc[-1]
                            if new_hrv != last_hrv and new_hrv > 0:
                                state.current_hrv = new_hrv
                                last_hrv = new_hrv
                            break
                    
                    if count % 20 == 0:
                        available_cols = list(data.columns)
                        print_status(f"CSV columns available: {available_cols}")
                        
                else:
                    print_status("CSV file empty")
            else:
                # Create dummy data for testing when no CSV file exists
                if count == 1:
                    print_status("output.csv not found - generating dummy HR and HRV data for testing")
                
                # Simulate realistic heart rate progression
                base_hr = 70
                base_hrv = 35  # Typical baseline RMSSD
                
                if state.warm_up_start and not state.baseline_complete:
                    # Baseline data
                    dummy_hr = base_hr + random.uniform(-5, 5)
                    dummy_hrv = base_hrv + random.uniform(-8, 8)
                elif state.warm_up_start:
                    # Warm-up data
                    dummy_hr = base_hr + random.uniform(10, 30)
                    dummy_hrv = base_hrv * random.uniform(0.7, 0.9)  # HRV decreases with activity
                elif state.current_exercise:
                    # Exercise-specific simulation
                    exercise_intensity = EXERCISE_CONFIG[state.current_exercise]["intensity_modifier"]
                    hrv_sensitivity = EXERCISE_CONFIG[state.current_exercise]["hrv_modifier"]
                    
                    target_hr_increase = 50 * exercise_intensity
                    dummy_hr = base_hr + target_hr_increase + random.uniform(-10, 15)
                    
                    # HRV decreases more with higher intensity exercises
                    hrv_reduction = (1.0 - exercise_intensity) * hrv_sensitivity
                    dummy_hrv = base_hrv * hrv_reduction * random.uniform(0.8, 1.2)
                else:
                    dummy_hr = base_hr + random.uniform(-5, 20)
                    dummy_hrv = base_hrv + random.uniform(-5, 5)
                
                state.mean_hr = max(50, dummy_hr)
                state.current_hrv = max(5, dummy_hrv)  # Ensure realistic minimum HRV
                
        except Exception as e:
            print_status(f"Data reading exception: {e}")

def main():
    """Main function to start the exercise system"""
    
    print_header("TERMINAL FITNESS COACH WITH HRV INTEGRATION")
    print("🏃‍♂️ Multi-Exercise Heart Rate & HRV Coach")
    print("🧠 Advanced HRV Safety Override System")
    print("💡 This is a test version for terminal use without robot hardware")
    print("\n📋 Instructions:")
    print("   • Make sure output.csv exists with 'mean_hr' and 'rmssd' columns")
    print("   • If no CSV, realistic dummy data will be generated for testing")
    print("   • HRV monitoring provides additional safety layer beyond heart rate")
    print("   • Follow the prompts and respond with the requested input")
    print("   • Press Ctrl+C at any time to exit")
    print("\n🧠 HRV Safety Features:")
    print("   • Green Zone: >60% of baseline (normal autonomic function)")
    print("   • Yellow Zone: 30-60% of baseline (increasing stress)")
    print("   • Red Zone: 20-30% of baseline (high stress - slow down)")
    print("   • Critical: <20% of baseline (stop exercise immediately)")
    
    input("\n🚀 Press Enter to start your fitness session...")

    # Start background thread
    data_thread = Thread(target=read_data, daemon=True)
    data_thread.start()
    print_status("✓ Data reader started")

    # Initial greeting
    print_header("WELCOME TO YOUR FITNESS SESSION")
    print_coach_message("Hello! Glad to see you here! Today, I will guide you through a personalized exercise session.")
    print_coach_message("I have five different exercises for you to choose from: Squats, Lunges, High Knees, Push-Ups, and Burpees.")
    print_coach_message("We will begin with collecting baseline heart rate and HRV data, then a warm-up, followed by your chosen exercise.")
    print_coach_message("🧠 I'll monitor both your heart rate and heart rate variability to ensure your safety!")
    
    # Start main exercise logic
    feedback_thread = Thread(target=calc_feedback, daemon=True)
    feedback_thread.start()
    print_status("✓ All systems started")

    try:
        # Main loop
        while not state.done:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n👋 Session interrupted by user. Goodbye!")
        state.done = True

    # Cleanup
    print_header("SESSION ENDED")
    print_status("Shutting down...")
    
    # Wait for threads to finish
    data_thread.join(timeout=5)
    feedback_thread.join(timeout=5)
    
    print_status("✓ Multi-exercise session with HRV monitoring complete!")
    print("📊 Thank you for testing the HRV-enhanced Terminal Fitness Coach!")

if __name__ == '__main__':
    main()
