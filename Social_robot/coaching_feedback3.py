#!/usr/bin/env python3
"""
HRV-Optimized QT Robot Multi-Exercise Coach
Heart Rate and HRV Monitoring System with Adaptive Rest Periods
Combines robot interaction with advanced HRV recovery optimization
"""

import time
import random
from threading import Thread, Lock
import rospy
from std_msgs.msg import String
import pandas as pd
import simpleaudio
import os
import numpy as np
from datetime import datetime
from collections import deque

# User configuration
AGE = 35
MAX_THRESH = .80
UPPER_THIRD = 0.75
LOWER_THIRD = 0.65
MIN_THRESH = .40

# Speech management settings
SPEECH_DELAY = 3.0
GESTURE_DELAY = 1.5
EMOTION_DELAY = 1.0
FEEDBACK_COOLDOWN = 15.0

max_hr = 220 - AGE

# HRV Configuration
HRV_CONFIG = {
    "baseline_duration": 120,  # 2 minutes baseline collection
    "warmup_duration": 30,     # 30 seconds warm-up
    "exercise_duration": 60,   # 1 minute exercise sets
    "final_rest_duration": 120, # 2 minutes final rest
    "min_rest_duration": 30,   # Minimum rest between sets
    "max_rest_duration": 180,  # Maximum rest between sets (3 minutes)
    "recovery_threshold": 0.85, # RMSSD recovery target (85% of baseline)
    "window_size": 5,          # Rolling window for RMSSD smoothing
    "trend_lookback": 4,       # Data points for trend analysis
    "sampling_rate": 1,        # How often RMSSD is updated (seconds)
}

# Enhanced exercise configurations with HRV impact
EXERCISE_CONFIG = {
    "squats": {
        "name": "Squats",
        "instructions": "Stand with feet shoulder-width apart, lower your body as if sitting back into a chair, then return to standing.",
        "metronome_normal": 1.2,
        "metronome_fast": 0.8,
        "metronome_slow": 1.8,
        "intensity_modifier": 0.85,
        "duration": 60,
        "hrv_impact": "moderate",
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
        "hrv_impact": "moderate",
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
        "hrv_impact": "high",
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
        "hrv_impact": "low",
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
        "hrv_impact": "very_high",
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

# Enhanced state management with HRV tracking
class HRVEnhancedExerciseState:
    def __init__(self):
        self.lock = Lock()
        
        # Basic state
        self._rest_hr = None
        self._inst_hr = -1
        self._mean_hr = -1
        self._done = False
        self._current_exercise = None
        
        # HRV state
        self._baseline_rmssd = None
        self._current_rmssd = -1
        self._rmssd_history = deque(maxlen=HRV_CONFIG["window_size"] * 3)
        self._baseline_rmssd_history = deque(maxlen=60)
        
        # Exercise phases
        self.phase = "baseline"  # baseline, warmup, exercise1, adaptive_rest, exercise2, final_rest
        self.exercise_selected = False
        self.set_number = 0
        
        # Rest optimization
        self.rest_start_time = None
        self.optimal_rest_achieved = False
        self.rest_feedback_count = 0
        
        # Exercise state (keeping original variables)
        self.warm_up_start = True
        self.slow_down = False
        self.inc_speed = False
        self.first_time = True
        self.stop_metronome = False
        self.baseline_complete = False
        
        # Speech management
        self.last_speech_time = 0
        self.last_feedback_time = 0
        self.speech_active = False
    
    # HRV property methods
    @property
    def baseline_rmssd(self):
        with self.lock:
            return self._baseline_rmssd
    
    @baseline_rmssd.setter
    def baseline_rmssd(self, value):
        with self.lock:
            self._baseline_rmssd = value
    
    @property
    def current_rmssd(self):
        with self.lock:
            return self._current_rmssd
    
    @current_rmssd.setter
    def current_rmssd(self, value):
        with self.lock:
            self._current_rmssd = value
            if value > 0:
                self._rmssd_history.append(value)
    
    @property
    def rmssd_history(self):
        with self.lock:
            return list(self._rmssd_history)
    
    # Existing property methods
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
state = HRVEnhancedExerciseState()

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
    """Managed speech function with interruption handling and timing control"""
    current_time = time.time()
    
    if priority != "high" and (current_time - state.last_speech_time) < SPEECH_DELAY:
        print(f"[SPEECH SKIPPED] Too soon since last speech: {message}")
        return False
    
    if priority == "high":
        time.sleep(0.5)
    
    success = safe_publish(speech_say_pub, message, "Speech", SPEECH_DELAY)
    
    if success:
        state.last_speech_time = current_time
        state.speech_active = True
        
        words = len(message.split())
        estimated_duration = (words / 150.0) * 60.0
        
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

# HRV Analysis Functions
def get_windowed_rmssd(window_size=5, method="weighted"):
    """Get smoothed RMSSD using sliding window"""
    history = state.rmssd_history
    
    if len(history) < 2:
        return state.current_rmssd if state.current_rmssd > 0 else None
    
    recent_values = history[-window_size:] if len(history) >= window_size else history
    
    if method == "simple":
        return np.mean(recent_values)
    elif method == "weighted":
        weights = np.linspace(0.5, 1.0, len(recent_values))
        weights = weights / np.sum(weights)
        return np.average(recent_values, weights=weights)
    elif method == "median":
        return np.median(recent_values)
    else:
        return recent_values[-1]

def get_rmssd_trend(lookback=4):
    """Analyze if RMSSD is trending upward (recovering) or downward"""
    history = state.rmssd_history
    
    if len(history) < lookback:
        return "insufficient_data"
    
    recent_values = history[-lookback:]
    x = np.arange(len(recent_values))
    slope = np.polyfit(x, recent_values, 1)[0]
    
    if slope > 2:
        return "strongly_improving"
    elif slope > 0.5:
        return "improving"
    elif slope > -0.5:
        return "stable"
    elif slope > -2:
        return "declining"
    else:
        return "strongly_declining"

def assess_hrv_recovery():
    """Assess current HRV recovery status using windowed RMSSD"""
    if state.baseline_rmssd is None:
        return None
    
    windowed_rmssd = get_windowed_rmssd(window_size=5, method="weighted")
    
    if windowed_rmssd is None or windowed_rmssd <= 0:
        return None
    
    recovery_ratio = windowed_rmssd / state.baseline_rmssd
    recovery_percent = recovery_ratio * 100
    trend = get_rmssd_trend()
    
    if recovery_ratio >= HRV_CONFIG["recovery_threshold"]:
        status = "excellent"
    elif recovery_ratio >= 0.75:
        status = "good"
    elif recovery_ratio >= 0.60:
        status = "moderate"
    else:
        status = "poor"
    
    return {
        "ratio": recovery_ratio,
        "percent": recovery_percent,
        "status": status,
        "recovered": recovery_ratio >= HRV_CONFIG["recovery_threshold"],
        "windowed_rmssd": windowed_rmssd,
        "trend": trend
    }

def get_adaptive_rest_feedback_speech():
    """Generate robot speech for adaptive rest period"""
    recovery = assess_hrv_recovery()
    
    if recovery is None:
        return "I'm monitoring your recovery. Please keep breathing naturally."
    
    rest_duration = time.time() - state.rest_start_time
    percent = recovery["percent"]
    status = recovery["status"]
    trend = recovery.get("trend", "stable")
    
    if recovery["recovered"]:
        if trend == "strongly_improving":
            return f"Excellent recovery! Your nervous system has bounced back to {percent:.0f} percent of baseline. You're ready!"
        else:
            return f"Perfect! Your recovery has reached {percent:.0f} percent. You're ready for the next set!"
    
    elif status == "good":
        if trend in ["improving", "strongly_improving"]:
            return f"Good recovery progress at {percent:.0f} percent. Your HRV is trending upward nicely."
        elif trend == "stable":
            return f"Good recovery level at {percent:.0f} percent. Your system is stabilizing well."
        else:
            return f"Good recovery level at {percent:.0f} percent. Let's wait for the upward trend."
    
    elif status == "moderate":
        if trend in ["improving", "strongly_improving"]:
            return f"Moderate recovery at {percent:.0f} percent, but I see improvement! Keep resting."
        elif trend == "declining":
            return f"Recovery is at {percent:.0f} percent and plateauing. This is normal, be patient."
        else:
            return f"Moderate recovery at {percent:.0f} percent. Keep breathing deeply and relaxing."
    
    else:  # Poor recovery
        if rest_duration > 90:
            if trend in ["improving", "strongly_improving"]:
                return f"Slow but steady recovery at {percent:.0f} percent. I see improvement, that's good!"
            else:
                return f"Recovery is at {percent:.0f} percent. This is normal after intense exercise!"
        else:
            return f"Early recovery phase at {percent:.0f} percent. Your nervous system is resetting."

def collect_baseline_hrv():
    """Collect baseline heart rate and HRV data with robot interaction"""
    print("=== BASELINE HRV COLLECTION ===")
    state.phase = "baseline"
    
    robot_speak_managed("Now I will collect your baseline heart rate and heart rate variability data. This helps me optimize your rest periods.", "high")
    robot_show_emotion("QT/happy")
    robot_play_gesture("QT/happy")
    time.sleep(3)
    
    robot_speak_managed("Please sit comfortably and breathe naturally for the next two minutes. Try to stay relaxed and calm.", "normal")
    robot_show_emotion("QT/confused")  # Calm/focused emotion
    time.sleep(3)
    
    baseline_duration = HRV_CONFIG["baseline_duration"]
    start_time = time.time()
    last_progress_time = 0
    
    while time.time() - start_time < baseline_duration:
        elapsed = time.time() - start_time
        remaining = baseline_duration - elapsed
        
        # Progress updates every 30 seconds
        if elapsed - last_progress_time >= 30 and remaining > 30:
            progress = (elapsed / baseline_duration) * 100
            robot_speak_managed(f"Baseline collection is {progress:.0f} percent complete. Keep breathing naturally.", "normal")
            last_progress_time = elapsed
            
            if state.current_rmssd > 0:
                state._baseline_rmssd_history.append(state.current_rmssd)
        
        time.sleep(1)
    
    # Calculate baseline values
    if state.mean_hr > 0:
        state.rest_hr = state.mean_hr
        robot_speak_managed(f"Excellent! Your baseline heart rate is {state.rest_hr:.0f} beats per minute.", "normal")
    
    if len(state._baseline_rmssd_history) > 0:
        state.baseline_rmssd = np.mean(list(state._baseline_rmssd_history))
        baseline_std = np.std(list(state._baseline_rmssd_history))
        
        robot_speak_managed(f"Perfect! Your baseline heart rate variability is {state.baseline_rmssd:.0f} milliseconds.", "normal")
        
        # Provide HRV insight with robot expressions
        if state.baseline_rmssd > 50:
            robot_speak_managed("Your HRV indicates excellent recovery status. You're ready for a great workout!", "normal")
            robot_show_emotion("QT/excited")
            robot_play_gesture("QT/show_muscles")
        elif state.baseline_rmssd > 30:
            robot_speak_managed("Your HRV shows good readiness for exercise. Let's have a productive session!", "normal")
            robot_show_emotion("QT/happy")
            robot_play_gesture("QT/happy")
        else:
            robot_speak_managed("Your HRV suggests taking it easier today. We'll adjust the intensity accordingly.", "normal")
            robot_show_emotion("QT/confused")
            robot_play_gesture("QT/attention")
    else:
        robot_speak_managed("HRV data is not available, but we can still have a great workout using heart rate monitoring.", "normal")
        state.baseline_rmssd = 40  # Default value
    
    state.baseline_complete = True
    time.sleep(2)

def optimize_rest_period():
    """Manage the adaptive rest period between exercise sets with robot interaction"""
    print("=== ADAPTIVE REST OPTIMIZATION ===")
    state.phase = "adaptive_rest"
    
    robot_speak_managed("Great work on your first set! Now I'll monitor your heart rate variability to optimize your rest period.", "high")
    robot_show_emotion("QT/happy", exercise_context=True)
    robot_play_gesture("QT/happy", exercise_context=True)
    time.sleep(3)
    
    robot_speak_managed("I'm analyzing your nervous system recovery. Please breathe deeply and relax.", "normal")
    robot_show_emotion("QT/confused")  # Calm/focused
    
    state.rest_start_time = time.time()
    state.optimal_rest_achieved = False
    state.rest_feedback_count = 0
    
    min_rest = HRV_CONFIG["min_rest_duration"]
    max_rest = HRV_CONFIG["max_rest_duration"]
    last_feedback_time = 0
    
    while True:
        rest_duration = time.time() - state.rest_start_time
        recovery = assess_hrv_recovery()
        current_time = time.time()
        
        # Provide feedback every 15 seconds
        if current_time - last_feedback_time >= 15:
            feedback_speech = get_adaptive_rest_feedback_speech()
            robot_speak_managed(feedback_speech, "normal")
            
            # Show appropriate emotions based on recovery status
            if recovery:
                if recovery["recovered"]:
                    robot_show_emotion("QT/excited")
                    robot_play_gesture("QT/show_muscles")
                elif recovery["status"] in ["good", "moderate"]:
                    robot_show_emotion("QT/happy")
                else:
                    robot_show_emotion("QT/confused")  # Patient/waiting
            
            last_feedback_time = current_time
        
        # Check if optimal rest is achieved
        if rest_duration >= min_rest:
            if recovery and recovery["recovered"]:
                state.optimal_rest_achieved = True
                robot_speak_managed(f"Perfect! Your HRV has recovered to {recovery['percent']:.0f} percent of baseline. That's optimal recovery!", "high")
                robot_show_emotion("QT/excited")
                robot_play_gesture("QT/excited")
                time.sleep(2)
                robot_speak_managed(f"Your rest period was {rest_duration:.0f} seconds. That's personalized to your recovery!", "normal")
                break
        
        # Check maximum rest duration
        if rest_duration >= max_rest:
            robot_speak_managed(f"We've reached the maximum rest time of {max_rest} seconds. Time for your second set!", "normal")
            if recovery:
                robot_speak_managed(f"Your recovery level is {recovery['percent']:.0f} percent, which is sufficient for continuing.", "normal")
            robot_show_emotion("QT/happy")
            robot_play_gesture("QT/happy")
            break
        
        time.sleep(1)

def select_exercise():
    """Enhanced exercise selection with HRV impact information"""
    print("=== EXERCISE SELECTION ===")
    
    exercises = list(EXERCISE_CONFIG.keys())
    
    robot_speak_managed("Great! Now let's choose your exercise. I have five options, each with different impacts on your recovery.", "high")
    robot_show_emotion("QT/happy")
    robot_play_gesture("QT/happy")
    time.sleep(2)
    
    for i, exercise_key in enumerate(exercises, 1):
        exercise_name = EXERCISE_CONFIG[exercise_key]["name"]
        hrv_impact = EXERCISE_CONFIG[exercise_key]["hrv_impact"]
        
        # Describe HRV impact in user-friendly terms
        impact_description = {
            "low": "gentle on your recovery",
            "moderate": "moderately challenging for recovery", 
            "high": "challenging for your nervous system",
            "very_high": "very demanding on recovery"
        }
        
        robot_speak_managed(f"Option {i}: {exercise_name}. This exercise is {impact_description.get(hrv_impact, 'moderate')}.", "normal")
        time.sleep(2)
    
    robot_speak_managed("Please tell me which number you'd like to choose, from 1 to 5.", "normal")
    
    print("Waiting for user selection...")
    time.sleep(8)
    
    # Simulate user selection (replace with actual input mechanism)
    selected_index = random.randint(0, len(exercises) - 1)
    selected_exercise = exercises[selected_index]
    
    exercise_name = EXERCISE_CONFIG[selected_exercise]["name"]
    instructions = EXERCISE_CONFIG[selected_exercise]["instructions"]
    hrv_impact = EXERCISE_CONFIG[selected_exercise]["hrv_impact"]
    
    robot_speak_managed(f"Excellent choice! You selected {exercise_name}.", "normal")
    time.sleep(2)
    
    robot_speak_managed(f"Here's how to do {exercise_name}: {instructions}", "normal")
    time.sleep(3)
    
    # Provide HRV-specific guidance
    if hrv_impact in ["high", "very_high"]:
        robot_speak_managed("This exercise is quite demanding, so I'll be extra careful with your rest periods.", "normal")
    else:
        robot_speak_managed("This exercise allows for good recovery, perfect for our HRV optimization!", "normal")
    
    robot_show_emotion("QT/happy", exercise_context=True)
    robot_play_gesture("QT/happy", exercise_context=True)
    
    state.current_exercise = selected_exercise
    state.exercise_selected = True
    
    print(f"Selected exercise: {exercise_name} (HRV impact: {hrv_impact})")
    time.sleep(3)
    return selected_exercise

def run_exercise_set(set_number):
    """Run a single exercise set with HRV monitoring"""
    current_exercise = state.current_exercise
    if not current_exercise:
        return
        
    exercise_config = EXERCISE_CONFIG[current_exercise]
    exercise_name = exercise_config["name"]
    exercise_duration = HRV_CONFIG["exercise_duration"]
    
    print(f"=== EXERCISE SET {set_number}: {exercise_name.upper()} ===")
    state.phase = f"exercise{set_number}"
    state.set_number = set_number
    
    robot_speak_managed(f"Set {set_number}: Let's do {exercise_name} for {exercise_duration} seconds!", "high")
    robot_show_emotion("QT/excited", exercise_context=True)
    robot_play_gesture("QT/excited", exercise_context=True)
    time.sleep(2)
    
    robot_speak_managed("I'm starting the metronome to help with your pacing. Let's go!", "normal")
    time.sleep(2)
    
    start_time = time.time()
    last_encouragement = 0
    
    while time.time() - start_time < exercise_duration:
        remaining = exercise_duration - (time.time() - start_time)
        elapsed = time.time() - start_time
        
        # Provide encouragement every 20 seconds
        if elapsed - last_encouragement >= 20 and remaining > 10:
            motivational_phrases = exercise_config["motivational_phrases"]
            encouragement = random.choice(motivational_phrases)
            robot_speak_managed(encouragement, "normal")
            robot_show_emotion(exercise_config["encouraging_emotion"])
            robot_play_gesture(exercise_config["encouraging_gesture"])
            last_encouragement = elapsed
        
        # Countdown for last 10 seconds
        if 5 < remaining <= 10 and int(remaining) == remaining:
            robot_speak_managed(f"{int(remaining)} seconds left!", "normal")
        
        time.sleep(1)
    
    robot_speak_managed(f"Set {set_number} complete! Excellent work!", "high")
    robot_show_emotion("QT/happy", exercise_context=True)
    robot_play_gesture("QT/show_muscles", exercise_context=True)
    time.sleep(2)

def final_rest_and_analysis():
    """Final rest period with comprehensive recovery analysis"""
    print("=== FINAL RECOVERY ANALYSIS ===")
    state.phase = "final_rest"
    
    robot_speak_managed("Excellent work! Now let's monitor your complete recovery for the next two minutes.", "high")
    robot_show_emotion("QT/happy")
    robot_play_gesture("QT/happy")
    time.sleep(3)
    
    robot_speak_managed("This final analysis will show how well your nervous system recovers from the complete workout.", "normal")
    
    final_duration = HRV_CONFIG["final_rest_duration"]
    start_time = time.time()
    recovery_data = []
    last_analysis_time = 0
    
    while time.time() - start_time < final_duration:
        elapsed = time.time() - start_time
        remaining = final_duration - elapsed
        current_time = time.time()
        
        # Analysis every 30 seconds
        if current_time - last_analysis_time >= 30 and remaining > 30:
            recovery = assess_hrv_recovery()
            if recovery:
                recovery_data.append(recovery["percent"])
                
                if recovery["percent"] > 100:
                    robot_speak_managed("Amazing! Your HRV is now above baseline. That's supercompensation!", "normal")
                    robot_show_emotion("QT/excited")
                    robot_play_gesture("QT/excited")
                elif recovery["recovered"]:
                    robot_speak_managed(f"Excellent recovery! Your HRV is at {recovery['percent']:.0f} percent of baseline.", "normal")
                    robot_show_emotion("QT/happy")
                else:
                    robot_speak_managed(f"Good progress! Recovery is at {recovery['percent']:.0f} percent and continuing.", "normal")
                    robot_show_emotion("QT/confused")  # Patient/monitoring
                    
                last_analysis_time = current_time
        
        # Progress update every minute
        if int(elapsed) % 60 == 0 and elapsed > 0 and remaining > 0:
            minutes_remaining = int(remaining / 60)
            robot_speak_managed(f"Final recovery monitoring: {minutes_remaining} minute remaining.", "normal")
        
        time.sleep(5)
    
    # Final comprehensive analysis
    if recovery_data:
        final_recovery = recovery_data[-1] if recovery_data else 0
        robot_speak_managed(f"Final recovery analysis complete! Your HRV recovered to {final_recovery:.0f} percent of baseline.", "high")
        
        if final_recovery >= 85:
            robot_speak_managed("Outstanding recovery! Your nervous system has fully bounced back. You could do another workout soon if you wanted!", "normal")
            robot_show_emotion("QT/excited")
            robot_play_gesture("QT/show_muscles")
        elif final_recovery >= 70:
            robot_speak_managed("Good recovery! Consider waiting a bit longer before your next intense session for optimal results.", "normal")
            robot_show_emotion("QT/happy")
            robot_play_gesture("QT/happy")
        else:
            robot_speak_managed("Take extra time to recover. Your body worked very hard today and deserves proper rest!", "normal")
            robot_show_emotion("QT/confused")
            robot_play_gesture("QT/attention")
    
    time.sleep(2)

def enhanced_hrv_session():
    """Run the complete HRV-optimized exercise session with robot interaction"""
    print("=== HRV-OPTIMIZED EXERCISE SESSION ===")
    
    # Phase 1: Baseline Collection
    collect_baseline_hrv()
    
    # Exercise Selection
    robot_speak_managed("Perfect! Now that I have your baseline data, let's select your exercise.", "normal")
    time.sleep(2)
    selected_exercise = select_exercise()
    
    robot_speak_managed("Make sure you have enough space around you and are ready to start.", "normal")
    robot_show_emotion("QT/happy")
    time.sleep(5)
    
    # Phase 2: Warm-up
    state.phase = "warmup"
    warmup_duration = HRV_CONFIG["warmup_duration"]
    
    robot_speak_managed(f"Let's start with a {warmup_duration}-second warm-up to prepare your body!", "high")
    robot_show_emotion("QT/happy", exercise_context=True)
    robot_play_gesture("QT/happy", exercise_context=True)
    
    warmup_start = time.time()
    while time.time() - warmup_start < warmup_duration:
        remaining = warmup_duration - (time.time() - warmup_start)
        if int(remaining) % 10 == 0 and remaining > 5:
            robot_speak_managed(f"Warm-up: {int(remaining)} seconds remaining. Keep moving gently!", "normal")
        time.sleep(5)
    
    robot_speak_managed("Warm-up complete! You should feel ready now.", "normal")
    time.sleep(2)
    
    # Phase 3: First Exercise Set
    run_exercise_set(set_number=1)
    
    # Phase 4: Adaptive Rest Period
    optimize_rest_period()
    
    robot_speak_managed("Your personalized rest period is complete. Ready for set two?", "normal")
    robot_show_emotion("QT/excited", exercise_context=True)
    robot_play_gesture("QT/excited", exercise_context=True)
    time.sleep(3)
    
    # Phase 5: Second Exercise Set
    run_exercise_set(set_number=2)
    
    # Phase 6: Final Rest and Analysis
    final_rest_and_analysis()
    
    # Session completion
    robot_speak_managed("Congratulations! You've completed your HRV-optimized workout session!", "high")
    robot_show_emotion("QT/excited")
    robot_play_gesture("QT/show_muscles")
    time.sleep(2)
    
    robot_speak_managed("Your personalized rest periods were based on your actual nervous system recovery. That's precision fitness!", "normal")
    robot_show_emotion("QT/happy")
    robot_play_gesture("QT/happy")

def play_metronome():
    """Play metronome with exercise-specific timing"""
    print("Starting HRV-optimized metronome...")
    
    try:
        strong_beat = simpleaudio.WaveObject.from_wave_file('strong_beat.wav')
        silence = simpleaudio.WaveObject.from_wave_file('silence_1.wav')
    except Exception as e:
        print(f"Audio files not found, using silent metronome: {e}")
        strong_beat = None
        silence = None

    while not state.done and not rospy.is_shutdown():
        try:
            current_exercise = state.current_exercise
            
            # Only play metronome during exercise phases
            if state.phase in ["exercise1", "exercise2"] and current_exercise:
                config = EXERCISE_CONFIG[current_exercise]
                normal_tempo = config["metronome_normal"]
                fast_tempo = config["metronome_fast"] 
                slow_tempo = config["metronome_slow"]
                
                if state.slow_down:
                    tempo = slow_tempo
                elif state.inc_speed:
                    tempo = fast_tempo
                else:
                    tempo = normal_tempo
                
                if strong_beat:
                    strong_beat.play()
                time.sleep(tempo)
            else:
                # Silent during rest periods
                if silence:
                    silence.play()
                time.sleep(0.5)
                
        except Exception as e:
            print(f"Metronome error: {e}")
            time.sleep(0.5)

def read_enhanced_data():
    """Enhanced data reader for HR and HRV with phase-based simulation"""
    print("Starting enhanced physiological data reader...")
    count = 0
    last_hr = -1
    last_rmssd = -1

    while not state.done and not rospy.is_shutdown():
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
                    
                    # Read RMSSD
                    if 'rmssd' in data.columns:
                        new_rmssd = data.rmssd.iloc[-1]
                        if new_rmssd != last_rmssd and new_rmssd > 0:
                            state.current_rmssd = new_rmssd
                            last_rmssd = new_rmssd
            else:
                # Enhanced dummy data simulation based on exercise phase
                if count == 1:
                    print("output.csv not found - generating phase-based dummy data")
                
                # Simulate realistic HR and HRV based on current phase
                if state.phase == "baseline":
                    dummy_hr = 70 + random.uniform(-5, 5)
                    dummy_rmssd = 45 + random.uniform(-10, 15)
                elif state.phase == "warmup":
                    dummy_hr = 85 + random.uniform(-5, 10)
                    dummy_rmssd = 35 + random.uniform(-8, 12)
                elif state.phase in ["exercise1", "exercise2"]:
                    # Higher intensity during exercise, lower HRV
                    exercise_config = EXERCISE_CONFIG.get(state.current_exercise, {})
                    hrv_impact = exercise_config.get("hrv_impact", "moderate")
                    
                    if hrv_impact == "very_high":
                        dummy_hr = 150 + random.uniform(-10, 20)
                        dummy_rmssd = 15 + random.uniform(-5, 8)
                    elif hrv_impact == "high":
                        dummy_hr = 140 + random.uniform(-10, 15)
                        dummy_rmssd = 20 + random.uniform(-8, 10)
                    elif hrv_impact == "moderate":
                        dummy_hr = 130 + random.uniform(-10, 15)
                        dummy_rmssd = 25 + random.uniform(-8, 10)
                    else:  # low impact
                        dummy_hr = 120 + random.uniform(-10, 15)
                        dummy_rmssd = 30 + random.uniform(-8, 10)
                        
                elif state.phase == "adaptive_rest":
                    # Simulate gradual recovery during adaptive rest
                    if state.rest_start_time:
                        rest_duration = time.time() - state.rest_start_time
                        recovery_factor = min(rest_duration / 90, 1.0)  # Recovery over 90 seconds
                        baseline_rmssd = state.baseline_rmssd or 45
                        
                        # Gradual HR decrease and HRV increase
                        dummy_hr = 120 - (40 * recovery_factor) + random.uniform(-8, 8)
                        dummy_rmssd = 20 + (baseline_rmssd - 20) * recovery_factor + random.uniform(-5, 5)
                    else:
                        dummy_hr = 100 + random.uniform(-10, 10)
                        dummy_rmssd = 30 + random.uniform(-8, 8)
                        
                elif state.phase == "final_rest":
                    # Continued recovery during final rest
                    dummy_hr = 75 + random.uniform(-5, 8)
                    baseline_rmssd = state.baseline_rmssd or 45
                    dummy_rmssd = baseline_rmssd * 0.85 + random.uniform(-8, 12)
                else:
                    # Default values
                    dummy_hr = 75 + random.uniform(-5, 10)
                    dummy_rmssd = 40 + random.uniform(-10, 10)
                
                state.mean_hr = max(50, dummy_hr)
                state.current_rmssd = max(10, dummy_rmssd)
                
        except Exception as e:
            print(f"Enhanced data reading error: {e}")

def main():
    """Main function with HRV-optimized exercise protocol"""
    global speech_say_pub, emotion_show_pub, gesture_play_pub, audio_play_pub
    
    print("=== QT Robot HRV-Optimized Multi-Exercise Coach ===")
    print("🤖 Robot-guided exercise with adaptive rest periods")
    print("💚 Heart Rate Variability optimization for peak performance")
    
    # Initialize ROS node
    try:
        rospy.init_node('qt_hrv_enhanced_coach', anonymous=True)
        print("✓ ROS node initialized")
        time.sleep(2)
        
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
        time.sleep(3)
        
    except Exception as e:
        print(f"✗ Failed to initialize publishers: {e}")
        return

    # Start background threads
    data_thread = Thread(target=read_enhanced_data, daemon=True)
    metronome_thread = Thread(target=play_metronome, daemon=True)

    data_thread.start()
    print("✓ Enhanced data reader started")

    # Initial greeting
    robot_speak_managed("Hello! I'm your HRV-optimized exercise coach. Today I'll use your heart rate variability to personalize your rest periods for maximum effectiveness.", "high")
    robot_show_emotion("QT/happy")
    robot_play_gesture("QT/happy")
    time.sleep(3)
    
    robot_speak_managed("This advanced approach ensures you get optimal recovery between exercise sets, leading to better performance and adaptation.", "normal")
    
    # Play welcome audio
    safe_publish(audio_play_pub, "QT/Komiku_Glouglou", "Audio")
    time.sleep(3)

    # Start metronome thread
    metronome_thread.start()
    print("✓ HRV-optimized metronome started")

    try:
        # Run the enhanced HRV session
        enhanced_hrv_session()
        
        # Final session completion message
        robot_speak_managed("Thank you for trying HRV-optimized training! Your personalized approach maximizes both performance and recovery.", "high")
        robot_show_emotion("QT/excited")
        robot_play_gesture("QT/show_muscles")
        
    except KeyboardInterrupt:
        print("\n=== Program interrupted by user ===")
        robot_speak_managed("Session interrupted. Thank you for exercising with me!", "high")
        state.done = True

    # Cleanup
    print("=== Shutting down HRV-optimized coach ===")
    state.done = True
    
    # Wait for threads to finish
    data_thread.join(timeout=10)
    metronome_thread.join(timeout=10)
    
    print("=== HRV-optimized exercise session complete ===")

if __name__ == '__main__':
    main()
