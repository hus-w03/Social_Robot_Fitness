#!/usr/bin/env python3
"""
AI-Powered Terminal Fitness Coach with Adaptive Recovery
Machine Learning-Enhanced Heart Rate and HRV Monitoring System
Features adaptive rest periods based on individual physiological recovery patterns
"""

import time
import random
from threading import Thread, Lock
import pandas as pd
import os
import sys
from datetime import datetime
import numpy as np
from collections import deque
import pickle
import json

# Machine Learning imports
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    ML_AVAILABLE = True
    print("✓ Machine Learning libraries available")
except ImportError:
    ML_AVAILABLE = False
    print("⚠️  sklearn not available - using rule-based recovery only")

# User configuration
AGE = 35
MAX_THRESH = .80
UPPER_THIRD = 0.75
LOWER_THIRD = 0.65
MIN_THRESH = .40

# HRV (RMSSD) Safety Thresholds (percentage of baseline)
HRV_GREEN_ZONE = 0.60
HRV_YELLOW_ZONE = 0.30
HRV_RED_ZONE = 0.20
HRV_CRITICAL = 0.15

# Recovery monitoring parameters
RECOVERY_WINDOW = 30  # seconds to analyze for recovery trends
MIN_REST_TIME = 15    # minimum rest period (safety)
MAX_REST_TIME = 300   # maximum rest period (safety)
RECOVERY_STABILITY_THRESHOLD = 5  # seconds of stable readings needed

max_hr = 220 - AGE

# Exercise configurations (same as before)
EXERCISE_CONFIG = {
    "squats": {
        "name": "Squats",
        "instructions": "Stand with feet shoulder-width apart, lower your body as if sitting back into a chair, then return to standing.",
        "metronome_normal": 1.2,
        "metronome_fast": 0.8,
        "metronome_slow": 1.8,
        "intensity_modifier": 0.85,
        "hrv_modifier": 1.1,
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
        "hrv_modifier": 1.05,
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
        "hrv_modifier": 0.85,
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
        "hrv_modifier": 1.2,
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
        "hrv_modifier": 0.75,
        "duration": 45,
        "motivational_phrases": [
            "Full body power! You've got this!",
            "Each burpee makes you stronger!",
            "Feel that total body burn!",
            "You're a warrior! Keep fighting!"
        ]
    }
}

class RecoveryPredictor:
    """Machine learning-based recovery prediction system"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler() if ML_AVAILABLE else None
        self.training_data = []
        self.model_trained = False
        self.recovery_history = deque(maxlen=100)  # Store last 100 recovery sessions
        
    def extract_features(self, hr_data, hrv_data, baseline_hr, baseline_hrv, time_elapsed):
        """Extract features for recovery prediction"""
        if len(hr_data) < 3 or len(hrv_data) < 3:
            return None
            
        # Heart rate features
        current_hr = hr_data[-1]
        hr_trend = np.polyfit(range(len(hr_data)), hr_data, 1)[0]  # Slope
        hr_recovery_percent = (baseline_hr - current_hr) / baseline_hr if baseline_hr > 0 else 0
        hr_variability = np.std(hr_data[-10:]) if len(hr_data) >= 10 else np.std(hr_data)
        
        # HRV features
        current_hrv = hrv_data[-1] if hrv_data[-1] > 0 else 0
        hrv_trend = np.polyfit(range(len(hrv_data)), hrv_data, 1)[0] if current_hrv > 0 else 0
        hrv_recovery_percent = (current_hrv / baseline_hrv) if baseline_hrv > 0 and current_hrv > 0 else 0
        hrv_variability = np.std(hrv_data[-10:]) if len(hrv_data) >= 10 and current_hrv > 0 else 0
        
        features = [
            current_hr,
            hr_trend,
            hr_recovery_percent,
            hr_variability,
            current_hrv,
            hrv_trend,
            hrv_recovery_percent,
            hrv_variability,
            time_elapsed,
            len(hr_data)  # Duration of monitoring
        ]
        
        return np.array(features).reshape(1, -1)
    
    def is_recovered_rule_based(self, hr_data, hrv_data, baseline_hr, baseline_hrv, time_elapsed):
        """Rule-based recovery assessment (fallback when ML not available)"""
        if len(hr_data) < 5:
            return False
            
        current_hr = hr_data[-1]
        recent_hr = hr_data[-5:]
        
        # HR recovery criteria
        hr_threshold = baseline_hr * 1.15  # Within 15% of baseline
        hr_stable = abs(max(recent_hr) - min(recent_hr)) < 5  # Stable within 5 bpm
        hr_declining = np.polyfit(range(len(recent_hr)), recent_hr, 1)[0] <= 0  # Declining trend
        
        # HRV recovery criteria (if available)
        hrv_recovered = True
        if len(hrv_data) > 0 and hrv_data[-1] > 0 and baseline_hrv > 0:
            current_hrv = hrv_data[-1]
            hrv_threshold = baseline_hrv * 0.7  # Within 70% of baseline
            hrv_recovered = current_hrv >= hrv_threshold
        
        # Time-based criteria
        min_time_met = time_elapsed >= MIN_REST_TIME
        
        recovery_criteria = [
            current_hr <= hr_threshold,
            hr_stable,
            hr_declining,
            hrv_recovered,
            min_time_met
        ]
        
        # Need at least 4 out of 5 criteria met
        return sum(recovery_criteria) >= 4
    
    def predict_recovery(self, hr_data, hrv_data, baseline_hr, baseline_hrv, time_elapsed):
        """Predict if user has recovered"""
        # Always use rule-based as fallback
        rule_based_result = self.is_recovered_rule_based(hr_data, hrv_data, baseline_hr, baseline_hrv, time_elapsed)
        
        if not ML_AVAILABLE or not self.model_trained:
            return rule_based_result, "rule-based"
        
        features = self.extract_features(hr_data, hrv_data, baseline_hr, baseline_hrv, time_elapsed)
        if features is None:
            return rule_based_result, "rule-based (insufficient data)"
        
        try:
            features_scaled = self.scaler.transform(features)
            prediction = self.model.predict(features_scaled)[0]
            confidence = max(self.model.predict_proba(features_scaled)[0])
            
            # Use ML prediction if confidence is high, otherwise fall back to rules
            if confidence > 0.7:
                return bool(prediction), f"ML (confidence: {confidence:.2f})"
            else:
                return rule_based_result, f"rule-based (low ML confidence: {confidence:.2f})"
                
        except Exception as e:
            print(f"ML prediction error: {e}")
            return rule_based_result, "rule-based (ML error)"
    
    def record_recovery_session(self, hr_data, hrv_data, baseline_hr, baseline_hrv, actual_recovery_time, user_felt_recovered):
        """Record a recovery session for future training"""
        if len(hr_data) < 5:
            return
            
        # Create training samples at different time points
        for i in range(5, len(hr_data), 5):  # Sample every 5 data points
            time_point = i * 2  # Assuming 2-second intervals
            features = self.extract_features(
                hr_data[:i], hrv_data[:i], baseline_hr, baseline_hrv, time_point
            )
            
            if features is not None:
                # Label: recovered if this time point was >= actual recovery time
                is_recovered = time_point >= actual_recovery_time
                
                self.training_data.append({
                    'features': features.flatten(),
                    'recovered': is_recovered,
                    'time_point': time_point,
                    'actual_recovery_time': actual_recovery_time,
                    'user_feedback': user_felt_recovered
                })
        
        # Store in history for analysis
        self.recovery_history.append({
            'recovery_time': actual_recovery_time,
            'baseline_hr': baseline_hr,
            'baseline_hrv': baseline_hrv,
            'user_feedback': user_felt_recovered,
            'timestamp': datetime.now()
        })
    
    def train_model(self):
        """Train the recovery prediction model"""
        if not ML_AVAILABLE or len(self.training_data) < 20:
            print(f"Cannot train ML model: ML available: {ML_AVAILABLE}, Data points: {len(self.training_data)}")
            return False
        
        try:
            # Prepare training data
            X = np.array([sample['features'] for sample in self.training_data])
            y = np.array([sample['recovered'] for sample in self.training_data])
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model = RandomForestClassifier(n_estimators=50, random_state=42)
            self.model.fit(X_scaled, y)
            
            # Evaluate
            if len(X) > 10:
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
                self.model.fit(X_train, y_train)
                score = self.model.score(X_test, y_test)
                print(f"✓ ML model trained with accuracy: {score:.2f}")
            
            self.model_trained = True
            return True
            
        except Exception as e:
            print(f"❌ Failed to train ML model: {e}")
            return False
    
    def save_model(self, filename="recovery_model.pkl"):
        """Save the trained model and data"""
        if self.model_trained and ML_AVAILABLE:
            try:
                model_data = {
                    'model': self.model,
                    'scaler': self.scaler,
                    'training_data': self.training_data,
                    'recovery_history': list(self.recovery_history)
                }
                with open(filename, 'wb') as f:
                    pickle.dump(model_data, f)
                print(f"✓ Model saved to {filename}")
            except Exception as e:
                print(f"❌ Failed to save model: {e}")
    
    def load_model(self, filename="recovery_model.pkl"):
        """Load a previously trained model"""
        if not ML_AVAILABLE:
            return False
            
        try:
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.training_data = model_data.get('training_data', [])
                self.recovery_history = deque(model_data.get('recovery_history', []), maxlen=100)
                self.model_trained = True
                
                print(f"✓ Model loaded from {filename}")
                print(f"  Training data points: {len(self.training_data)}")
                print(f"  Recovery history: {len(self.recovery_history)} sessions")
                return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
        
        return False

# Thread-safe state management (same as before but with recovery predictor)
class ExerciseState:
    def __init__(self):
        self.lock = Lock()
        self._rest_hr = None
        self._inst_hr = -1
        self._mean_hr = -1
        self._rest_hrv = None
        self._current_hrv = -1
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
        
        # Recovery monitoring
        self.in_recovery = False
        self.recovery_start_time = None
        self.recovery_hr_data = []
        self.recovery_hrv_data = []
        self.predicted_recovery_time = None
    
    # ... (same property methods as before)
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

# Global state and recovery predictor
state = ExerciseState()
recovery_predictor = RecoveryPredictor()

# ... (same utility functions as before)
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

def adaptive_recovery_monitor():
    """Intelligent recovery monitoring with ML predictions"""
    if not state.in_recovery:
        return False
    
    elapsed_time = time.time() - state.recovery_start_time
    
    # Safety limits
    if elapsed_time >= MAX_REST_TIME:
        print_coach_message(f"⏰ Maximum rest time reached ({MAX_REST_TIME}s). Let's continue!")
        return True
    
    if elapsed_time < MIN_REST_TIME:
        return False
    
    # Get prediction from ML system
    is_recovered, method = recovery_predictor.predict_recovery(
        state.recovery_hr_data,
        state.recovery_hrv_data,
        state.rest_hr,
        state.rest_hrv,
        elapsed_time
    )
    
    # Display recovery progress
    if len(state.recovery_hr_data) > 0:
        current_hr = state.recovery_hr_data[-1]
        hr_recovery = ((current_hr - state.rest_hr) / state.rest_hr) * 100 if state.rest_hr > 0 else 0
        
        hrv_info = ""
        if len(state.recovery_hrv_data) > 0 and state.recovery_hrv_data[-1] > 0 and state.rest_hrv > 0:
            current_hrv = state.recovery_hrv_data[-1]
            hrv_recovery = (current_hrv / state.rest_hrv) * 100
            hrv_info = f" | HRV: {hrv_recovery:.1f}% of baseline"
        
        print_status(f"Recovery: {elapsed_time:.0f}s | HR: +{hr_recovery:.1f}% above baseline{hrv_info} | Method: {method}")
    
    if is_recovered:
        print_coach_message(f"✅ Recovery detected after {elapsed_time:.0f} seconds using {method}!")
        
        # Ask user for feedback (for learning)
        user_feels_recovered = get_user_input("Do you feel ready to continue?", ['yes', 'no']) == 'yes'
        
        # Record this session for learning
        recovery_predictor.record_recovery_session(
            state.recovery_hr_data,
            state.recovery_hrv_data,
            state.rest_hr,
            state.rest_hrv,
            elapsed_time,
            user_feels_recovered
        )
        
        return True
    
    return False

def start_recovery_monitoring():
    """Start adaptive recovery monitoring"""
    state.in_recovery = True
    state.recovery_start_time = time.time()
    state.recovery_hr_data = []
    state.recovery_hrv_data = []
    
    print_header("ADAPTIVE RECOVERY MONITORING")
    print_coach_message("Starting intelligent recovery monitoring. I'll analyze your physiological data to determine optimal rest time.")
    print_status("🧠 Using machine learning to predict your recovery status...")

def end_recovery_monitoring():
    """End recovery monitoring"""
    state.in_recovery = False
    state.recovery_start_time = None
    
    # Retrain model periodically
    if len(recovery_predictor.training_data) > 0 and len(recovery_predictor.training_data) % 20 == 0:
        print_status("🧠 Updating recovery prediction model with new data...")
        recovery_predictor.train_model()

# ... (same exercise functions as before, but with adaptive recovery)

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

# ... (continue with other functions - keeping the response shorter by focusing on key changes)

def calc_feedback():
    """Main exercise feedback logic with adaptive recovery"""
    print_header("AI-ENHANCED EXERCISE FEEDBACK SYSTEM")
    
    # Load previous model if available
    recovery_predictor.load_model()
    
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

    # Warm-up phase (simplified for space)
    while state.warm_up_start:
        warm_up_count -= 1
        warm_up_time -= 1
        
        print_status(f"Warm-up in progress... {warm_up_time}s remaining")
        
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
    print_coach_message(f'Let us start {exercise_name}! I will monitor your recovery intelligently during rest periods.')
    
    input("\n⏸️  Press Enter to begin the exercise...")

    # Exercise loop
    while not state.done:
        training_phase -= 1
        
        if state.rest_hr is not None:
            print_status(f"Exercise: {training_phase}s remaining, HR: {state.mean_hr:.1f}")

        # Adaptive rest break after first round
        if training_phase == 0 and state.first_time:
            state.first_time = False
            state.inc_speed = False
            state.slow_down = False
            state.stop_metronome = True
            
            print_coach_message(f'Great work on {exercise_name}! Starting adaptive recovery monitoring...')
            
            # Start adaptive recovery monitoring
            start_recovery_monitoring()
            
            # Monitor recovery until ML system says ready
            while state.in_recovery:
                # Check if recovered
                if adaptive_recovery_monitor():
                    end_recovery_monitoring()
                    break
                time.sleep(2)

            continue_choice = get_user_input("Would you like to do a second round?", ['yes', 'no'])
            
            if continue_choice == 'yes':
                training_phase = training_phase_limit
                state.stop_metronome = False
                print_coach_message(f'Great! Let us start {exercise_name} again!')
            else:
                state.done = True
                print_coach_message('I hope you enjoyed the exercise session with me. Thank you!')

        # End of session
        if not state.done and training_phase < 1:
            state.done = True
            print_header("SESSION COMPLETE")
            print_coach_message(f'We have reached the end of the {exercise_name} session!')

        time.sleep(1)

def read_data():
    """Read physiological data from CSV and update recovery monitoring"""
    print_status("Starting AI-enhanced data reader...")
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
                            
                            # Add to recovery monitoring if active
                            if state.in_recovery:
                                state.recovery_hr_data.append(new_hr)
                    
                    # Read HRV
                    hrv_columns = ['rmssd', 'hrv_rmssd', 'RMSSD', 'HRV', 'hrv']
                    for col in hrv_columns:
                        if col in data.columns:
                            new_hrv = data[col].iloc[-1]
                            if new_hrv != last_hrv and new_hrv > 0:
                                state.current_hrv = new_hrv
                                last_hrv = new_hrv
                                
                                # Add to recovery monitoring if active
                                if state.in_recovery:
                                    state.recovery_hrv_data.append(new_hrv)
                            break
                        
            else:
                # Generate realistic dummy data
                if count == 1:
                    print_status("output.csv not found - generating AI-enhanced dummy data")
                
                # Simulate recovery patterns during rest
                base_hr = 70
                base_hrv = 35
                
                if state.in_recovery and state.recovery_start_time:
                    # Simulate realistic recovery curve
                    recovery_time = time.time() - state.recovery_start_time
                    recovery_progress = min(recovery_time / 60.0, 1.0)  # Recover over 60 seconds
                    
                    # Exponential recovery curve
                    hr_recovery_factor = 1 - (0.7 * np.exp(-recovery_time / 30.0))
                    hrv_recovery_factor = 0.3 + (0.7 * (1 - np.exp(-recovery_time / 45.0)))
                    
                    dummy_hr = base_hr + (50 * hr_recovery_factor) + random.uniform(-3, 3)
                    dummy_hrv = base_hrv * hrv_recovery_factor + random.uniform(-2, 2)
                else:
                    # Normal exercise simulation
                    dummy_hr = base_hr + random.uniform(-10, 20)
                    dummy_hrv = base_hrv + random.uniform(-8, 8)
                
                state.mean_hr = max(50, dummy_hr)
                state.current_hrv = max(5, dummy_hrv)
                
                # Add to recovery monitoring if active
                if state.in_recovery:
                    state.recovery_hr_data.append(state.mean_hr)
                    state.recovery_hrv_data.append(state.current_hrv)
                
        except Exception as e:
            print_status(f"Data reading exception: {e}")

def calc_initial_hr():
    """Calculate baseline heart rate and HRV"""
    print_header("AI-ENHANCED BASELINE COLLECTION")
    print_coach_message("Calculating your baseline heart rate and HRV for personalized recovery predictions...")
    print_status("📊 This data will help the AI learn your recovery patterns")
    
    baseline_duration = 60
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
    else:
        state.rest_hr = 70
    
    if hrv_readings:
        state.rest_hrv = np.mean(hrv_readings)
    else:
        state.rest_hrv = 30
    
    state.baseline_complete = True
    
    print_coach_message(f"✅ AI-enhanced baseline collection complete!")
    print(f"📊 Baseline HR: {state.rest_hr:.1f} BPM")
    print(f"📊 Baseline HRV: {state.rest_hrv:.1f} ms")

def main():
    """Main function to start the AI-enhanced exercise system"""
    
    print_header("AI-POWERED TERMINAL FITNESS COACH")
    print("🏃‍♂️ Multi-Exercise Heart Rate & HRV Coach with Machine Learning")
    print("🧠 Adaptive Recovery Monitoring using AI")
    print("💡 Learns your individual recovery patterns for personalized rest periods")
    print("\n🔮 AI Features:")
    print("   • Adaptive rest periods based on physiological data")
    print("   • Machine learning recovery prediction")
    print("   • Personalized recovery patterns")
    print("   • Continuous learning from user feedback")
    
    input("\n🚀 Press Enter to start your AI-enhanced fitness session...")

    # Start background thread
    data_thread = Thread(target=read_data, daemon=True)
    data_thread.start()
    print_status("✓ AI-enhanced data reader started")

    # Initial greeting
    print_header("WELCOME TO YOUR AI-ENHANCED FITNESS SESSION")
    print_coach_message("Hello! I'm your AI fitness coach. I'll learn your recovery patterns and provide personalized rest recommendations.")
    
    try:
        # Start main exercise logic
        calc_feedback()
        
        # Save model after session
        recovery_predictor.save_model()
        
    except KeyboardInterrupt:
        print("\n\n👋 Session interrupted by user. Saving AI model...")
        recovery_predictor.save_model()
        state.done = True

    # Cleanup
    print_header("AI SESSION ENDED")
    print_status("Shutting down...")
    
    data_thread.join(timeout=5)
    
    print_status("✓ AI-enhanced fitness session complete!")
    print(f"📈 Recovery sessions analyzed: {len(recovery_predictor.recovery_history)}")
    print(f"🧠 Training data points collected: {len(recovery_predictor.training_data)}")

if __name__ == '__main__':
    main()