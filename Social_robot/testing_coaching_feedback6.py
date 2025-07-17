#!/usr/bin/env python3
"""
HRV-Optimized Exercise Coach with Adaptive Rest Periods
Uses RMSSD values to optimize rest periods between exercise sets
"""

import time
import random
from threading import Thread, Lock
import pandas as pd
import os
import sys
from datetime import datetime
from collections import deque
import numpy as np

# User configuration
AGE = 35
MAX_THRESH = .80
UPPER_THIRD = 0.75
LOWER_THIRD = 0.65
MIN_THRESH = .40

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

# Exercise configurations (keeping your existing structure)
EXERCISE_CONFIG = {
    "squats": {
        "name": "Squats",
        "instructions": "Stand with feet shoulder-width apart, lower your body as if sitting back into a chair, then return to standing.",
        "metronome_normal": 1.2,
        "metronome_fast": 0.8,
        "metronome_slow": 1.8,
        "intensity_modifier": 0.85,
        "hrv_impact": "moderate",  # How much this exercise typically impacts HRV
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
        "hrv_impact": "moderate",
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
        "hrv_impact": "high",  # High intensity cardio impacts HRV more
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
        "hrv_impact": "low",  # Strength exercises have lower HRV impact
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
        "hrv_impact": "very_high",  # Burpees have highest HRV impact
        "motivational_phrases": [
            "Full body power! You've got this!",
            "Each burpee makes you stronger!",
            "Feel that total body burn!",
            "You're a warrior! Keep fighting!"
        ]
    }
}

# Enhanced state management with HRV tracking
class EnhancedExerciseState:
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
        self._rmssd_history = deque(maxlen=HRV_CONFIG["window_size"] * 3)  # Store more for trend analysis
        self._baseline_rmssd_history = deque(maxlen=60)  # Store baseline values
        
        # Exercise phases
        self.phase = "baseline"  # baseline, warmup, exercise1, rest, exercise2, final_rest
        self.exercise_selected = False
        self.set_number = 0
        
        # Rest optimization
        self.rest_start_time = None
        self.optimal_rest_achieved = False
        self.rest_feedback_count = 0
        
        # Exercise state (keeping your existing variables)
        self.warm_up_start = True
        self.slow_down = False
        self.inc_speed = False
        self.first_time = True
        self.stop_metronome = False
    
    # Property methods for thread-safe access
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
    
    # Keep existing properties
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
state = EnhancedExerciseState()

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_coach_message(message):
    """Print coach message with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] 🤖 COACH: {message}")

def print_status(message):
    """Print status message"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] ℹ️  STATUS: {message}")

def print_hrv_status(rmssd, recovery_percent=None, show_windowed=True):
    """Print HRV status with recovery information and windowed values"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    if show_windowed and len(state.rmssd_history) >= 2:
        windowed_rmssd = get_windowed_rmssd()
        if windowed_rmssd:
            if recovery_percent:
                print(f"[{timestamp}] 💚 HRV: {rmssd:.1f}ms (Smoothed: {windowed_rmssd:.1f}ms, Recovery: {recovery_percent:.1f}%)")
            else:
                print(f"[{timestamp}] 💚 HRV: {rmssd:.1f}ms (Smoothed: {windowed_rmssd:.1f}ms)")
        else:
            if recovery_percent:
                print(f"[{timestamp}] 💚 HRV: {rmssd:.1f}ms (Recovery: {recovery_percent:.1f}%)")
            else:
                print(f"[{timestamp}] 💚 HRV: {rmssd:.1f}ms")
    else:
        if recovery_percent:
            print(f"[{timestamp}] 💚 HRV: {rmssd:.1f}ms (Recovery: {recovery_percent:.1f}%)")
        else:
            print(f"[{timestamp}] 💚 HRV: {rmssd:.1f}ms")

def get_windowed_rmssd(window_size=5, method="weighted"):
    """Get smoothed RMSSD using sliding window"""
    history = state.rmssd_history
    
    if len(history) < 2:
        return state.current_rmssd if state.current_rmssd > 0 else None
    
    # Use only the most recent values up to window_size
    recent_values = history[-window_size:] if len(history) >= window_size else history
    
    if method == "simple":
        # Simple moving average
        return np.mean(recent_values)
    
    elif method == "weighted":
        # Weighted average (more recent values have higher weight)
        weights = np.linspace(0.5, 1.0, len(recent_values))
        weights = weights / np.sum(weights)  # Normalize
        return np.average(recent_values, weights=weights)
    
    elif method == "median":
        # Median filter (robust to outliers)
        return np.median(recent_values)
    
    else:
        return recent_values[-1]  # Just return most recent

def assess_hrv_recovery():
    """Assess current HRV recovery status using windowed RMSSD"""
    if state.baseline_rmssd is None:
        return None
    
    # Get smoothed RMSSD value
    windowed_rmssd = get_windowed_rmssd(window_size=5, method="weighted")
    
    if windowed_rmssd is None or windowed_rmssd <= 0:
        return None
    
    recovery_ratio = windowed_rmssd / state.baseline_rmssd
    recovery_percent = recovery_ratio * 100
    
    # Also track trend direction
    trend = get_rmssd_trend()
    
    # Classify recovery status
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

def get_rmssd_trend(lookback=4):
    """Analyze if RMSSD is trending upward (recovering) or downward"""
    history = state.rmssd_history
    
    if len(history) < lookback:
        return "insufficient_data"
    
    recent_values = history[-lookback:]
    
    # Simple linear trend analysis
    x = np.arange(len(recent_values))
    slope = np.polyfit(x, recent_values, 1)[0]
    
    # Classify trend
    if slope > 2:  # Strong upward trend
        return "strongly_improving"
    elif slope > 0.5:  # Moderate upward trend
        return "improving"
    elif slope > -0.5:  # Stable
        return "stable"
    elif slope > -2:  # Moderate downward
        return "declining"
    else:  # Strong downward
        return "strongly_declining"

def get_adaptive_rest_feedback():
    """Provide feedback during adaptive rest period using windowed RMSSD and trends"""
    recovery = assess_hrv_recovery()
    
    if recovery is None:
        return "Monitoring your recovery..."
    
    rest_duration = time.time() - state.rest_start_time
    percent = recovery["percent"]
    status = recovery["status"]
    trend = recovery.get("trend", "stable")
    
    # Enhanced feedback incorporating trend information
    if recovery["recovered"]:
        if trend == "strongly_improving":
            return f"✅ Excellent recovery! ({percent:.1f}%) Strong upward trend - you're ready!"
        else:
            return f"✅ Excellent recovery! ({percent:.1f}%) You're ready for the next set!"
    
    elif status == "good":
        if trend in ["improving", "strongly_improving"]:
            return f"🟡 Good recovery progress ({percent:.1f}%). Trending upward - almost ready..."
        elif trend == "stable":
            return f"🟡 Good recovery level ({percent:.1f}%). Stabilizing nicely..."
        else:
            return f"🟡 Good recovery level ({percent:.1f}%), but let's wait for upward trend..."
    
    elif status == "moderate":
        if trend in ["improving", "strongly_improving"]:
            return f"🟠 Moderate recovery ({percent:.1f}%), but improving trend! Keep resting..."
        elif trend == "declining":
            return f"🟠 Recovery plateauing ({percent:.1f}%). This is normal - be patient..."
        else:
            return f"🟠 Moderate recovery ({percent:.1f}%). Keep resting..."
    
    else:  # Poor recovery
        if rest_duration > 90:  # If resting for a while with poor recovery
            if trend in ["improving", "strongly_improving"]:
                return f"🔴 Slow but steady recovery ({percent:.1f}%). I see improvement - good!"
            else:
                return f"🔴 Recovery taking time ({percent:.1f}%). This is normal after intense exercise!"
        else:
            return f"🔴 Early recovery phase ({percent:.1f}%). Your nervous system is resetting..."

def optimize_rest_period():
    """Manage the adaptive rest period between exercise sets"""
    print_header("ADAPTIVE REST PERIOD")
    print_coach_message("Great work on your first set! Now I'll monitor your HRV to optimize your rest.")
    print_status("Analyzing your nervous system recovery...")
    
    state.rest_start_time = time.time()
    state.optimal_rest_achieved = False
    state.rest_feedback_count = 0
    
    min_rest = HRV_CONFIG["min_rest_duration"]
    max_rest = HRV_CONFIG["max_rest_duration"]
    
    while True:
        rest_duration = time.time() - state.rest_start_time
        recovery = assess_hrv_recovery()
        
        # Provide periodic feedback
        if state.rest_feedback_count % 10 == 0:  # Every 10 seconds
            feedback = get_adaptive_rest_feedback()
            print_coach_message(feedback)
            
            if recovery:
                print_hrv_status(state.current_rmssd, recovery["percent"])
        
        state.rest_feedback_count += 1
        
        # Check if optimal rest is achieved
        if rest_duration >= min_rest:
            if recovery and recovery["recovered"]:
                state.optimal_rest_achieved = True
                print_coach_message(f"🎯 Perfect! Your HRV has recovered to {recovery['percent']:.1f}% of baseline.")
                print_coach_message(f"Rest duration: {rest_duration:.0f} seconds - optimally recovered!")
                break
        
        # Check maximum rest duration
        if rest_duration >= max_rest:
            print_coach_message(f"⏰ Maximum rest time reached ({max_rest}s). Time for your second set!")
            if recovery:
                print_coach_message(f"Recovery level: {recovery['percent']:.1f}% - that's sufficient for continuing.")
            break
        
        time.sleep(1)

def collect_baseline_hrv():
    """Collect baseline heart rate and HRV data"""
    print_header("BASELINE DATA COLLECTION")
    print_coach_message("Now I'll collect your baseline heart rate and HRV data.")
    print_status("Please sit comfortably and breathe naturally for 2 minutes...")
    
    baseline_duration = HRV_CONFIG["baseline_duration"]
    start_time = time.time()
    
    # Collect data for 2 minutes
    while time.time() - start_time < baseline_duration:
        elapsed = time.time() - start_time
        remaining = baseline_duration - elapsed
        
        # Show progress every 10 seconds
        if int(elapsed) % 10 == 0 and int(elapsed) > 0:
            progress = (elapsed / baseline_duration) * 100
            print_status(f"Baseline collection: {progress:.0f}% complete ({remaining:.0f}s remaining)")
            
            if state.mean_hr > 0:
                print(f"   Current HR: {state.mean_hr:.1f} BPM")
            if state.current_rmssd > 0:
                print(f"   Current HRV: {state.current_rmssd:.1f}ms")
                state._baseline_rmssd_history.append(state.current_rmssd)
        
        time.sleep(1)
    
    # Calculate baseline values
    if state.mean_hr > 0:
        state.rest_hr = state.mean_hr
        print_coach_message(f"✅ Baseline HR collected: {state.rest_hr:.1f} BPM")
    
    if len(state._baseline_rmssd_history) > 0:
        state.baseline_rmssd = np.mean(list(state._baseline_rmssd_history))
        baseline_std = np.std(list(state._baseline_rmssd_history))
        print_coach_message(f"✅ Baseline HRV collected: {state.baseline_rmssd:.1f} ± {baseline_std:.1f}ms")
        
        # Provide HRV insight
        if state.baseline_rmssd > 50:
            print_coach_message("Your HRV indicates excellent recovery status - ready for a good workout!")
        elif state.baseline_rmssd > 30:
            print_coach_message("Your HRV shows good readiness for exercise.")
        else:
            print_coach_message("Your HRV suggests taking it easier today - listen to your body.")
    else:
        print_coach_message("⚠️  HRV data not available. Proceeding with HR-only monitoring.")
        state.baseline_rmssd = 40  # Default value

def enhanced_exercise_session():
    """Run the complete HRV-optimized exercise session"""
    print_header("HRV-OPTIMIZED EXERCISE SESSION")
    
    # Phase 1: Baseline Collection (2 minutes)
    state.phase = "baseline"
    collect_baseline_hrv()
    
    # Exercise Selection
    selected_exercise = select_exercise()
    exercise_config = EXERCISE_CONFIG[selected_exercise]
    
    input("\n⏸️  Press Enter when you're ready to start the warm-up...")
    
    # Phase 2: Warm-up (30 seconds)
    state.phase = "warmup"
    warmup_duration = HRV_CONFIG["warmup_duration"]
    
    print_header("WARM-UP PHASE")
    print_coach_message(f"Let's start with a {warmup_duration}-second warm-up!")
    
    warmup_start = time.time()
    while time.time() - warmup_start < warmup_duration:
        remaining = warmup_duration - (time.time() - warmup_start)
        print_status(f"Warm-up: {remaining:.0f}s remaining")
        time.sleep(5)
    
    # Phase 3: First Exercise Set (1 minute)
    state.phase = "exercise1"
    state.set_number = 1
    run_exercise_set(exercise_config, set_number=1)
    
    # Phase 4: Adaptive Rest Period
    state.phase = "rest"
    optimize_rest_period()
    
    input("\n⏸️  Press Enter when you're ready for your second set...")
    
    # Phase 5: Second Exercise Set (1 minute)
    state.phase = "exercise2"
    state.set_number = 2
    run_exercise_set(exercise_config, set_number=2)
    
    # Phase 6: Final Rest and Data Collection (2 minutes)
    state.phase = "final_rest"
    final_rest_and_analysis()
    
    print_header("SESSION COMPLETE")
    print_coach_message("Congratulations! You've completed your HRV-optimized workout!")

def run_exercise_set(exercise_config, set_number):
    """Run a single exercise set with HRV monitoring"""
    exercise_name = exercise_config["name"]
    exercise_duration = HRV_CONFIG["exercise_duration"]
    
    print_header(f"EXERCISE SET {set_number}: {exercise_name.upper()}")
    print_coach_message(f"Set {set_number}: {exercise_name} for {exercise_duration} seconds!")
    
    # Start metronome
    metronome_thread = Thread(target=virtual_metronome, daemon=True)
    metronome_thread.start()
    
    start_time = time.time()
    
    while time.time() - start_time < exercise_duration:
        remaining = exercise_duration - (time.time() - start_time)
        
        # Show progress and vitals
        if int(remaining) % 10 == 0 and remaining > 0:
            print_status(f"Exercise: {remaining:.0f}s remaining")
            if state.mean_hr > 0:
                print(f"   Current HR: {state.mean_hr:.1f} BPM")
            if state.current_rmssd > 0:
                print(f"   Current HRV: {state.current_rmssd:.1f}ms")
        
        time.sleep(1)
    
    print_coach_message(f"✅ Set {set_number} complete! Great work!")

def final_rest_and_analysis():
    """Final rest period with data collection and analysis"""
    print_header("FINAL RECOVERY ANALYSIS")
    print_coach_message("Excellent work! Now let's monitor your recovery for 2 minutes.")
    
    final_duration = HRV_CONFIG["final_rest_duration"]
    start_time = time.time()
    
    recovery_data = []
    
    while time.time() - start_time < final_duration:
        elapsed = time.time() - start_time
        remaining = final_duration - elapsed
        
        if int(elapsed) % 15 == 0 and elapsed > 0:  # Every 15 seconds
            recovery = assess_hrv_recovery()
            if recovery:
                recovery_data.append(recovery["percent"])
                print_coach_message(f"Recovery analysis: {recovery['percent']:.1f}% of baseline HRV")
                
                if recovery["percent"] > 100:
                    print_coach_message("🌟 Supercompensation detected! Your HRV is above baseline!")
                elif recovery["recovered"]:
                    print_coach_message("✅ Excellent recovery! Your nervous system has bounced back.")
        
        if int(remaining) % 30 == 0 and remaining > 0:
            print_status(f"Final recovery monitoring: {remaining:.0f}s remaining")
        
        time.sleep(5)
    
    # Final analysis
    if recovery_data:
        final_recovery = recovery_data[-1] if recovery_data else 0
        print_coach_message(f"🎯 Final recovery status: {final_recovery:.1f}% of baseline")
        
        if final_recovery >= 85:
            print_coach_message("Outstanding recovery! You're ready for another workout soon.")
        elif final_recovery >= 70:
            print_coach_message("Good recovery. Consider waiting a bit longer before your next intense session.")
        else:
            print_coach_message("Take extra time to recover. Your body worked hard today!")

def select_exercise():
    """Exercise selection with HRV considerations"""
    print_header("EXERCISE SELECTION")
    
    exercises = list(EXERCISE_CONFIG.keys())
    
    print_coach_message("Let's choose your exercise. Here are your options:")
    
    for i, exercise_key in enumerate(exercises, 1):
        exercise_name = EXERCISE_CONFIG[exercise_key]["name"]
        hrv_impact = EXERCISE_CONFIG[exercise_key]["hrv_impact"]
        instructions = EXERCISE_CONFIG[exercise_key]["instructions"]
        
        impact_emoji = {"low": "🟢", "moderate": "🟡", "high": "🟠", "very_high": "🔴"}
        
        print(f"\n{i}. {exercise_name} {impact_emoji.get(hrv_impact, '🟡')}")
        print(f"   📝 {instructions}")
        print(f"   💚 HRV Impact: {hrv_impact}")
    
    while True:
        try:
            choice = input("\n🎤 Please enter the number of your chosen exercise (1-5): ").strip()
            choice_num = int(choice)
            
            if 1 <= choice_num <= 5:
                selected_exercise = exercises[choice_num - 1]
                exercise_name = EXERCISE_CONFIG[selected_exercise]["name"]
                
                print_coach_message(f"Excellent choice! You selected {exercise_name}.")
                
                state.current_exercise = selected_exercise
                state.exercise_selected = True
                
                return selected_exercise
            else:
                print("❌ Please enter a number between 1 and 5.")
                
        except ValueError:
            print("❌ Please enter a valid number.")

def virtual_metronome():
    """Visual metronome for exercise pacing"""
    while state.phase in ["exercise1", "exercise2"] and not state.done:
        current_exercise = state.current_exercise
        
        if current_exercise and current_exercise in EXERCISE_CONFIG:
            config = EXERCISE_CONFIG[current_exercise]
            tempo = config["metronome_normal"]
        else:
            tempo = 0.5
        
        print("🔴", end=" ", flush=True)
        time.sleep(tempo)

def read_physiological_data():
    """Enhanced data reader for HR and HRV"""
    print_status("Starting physiological data reader...")
    count = 0
    last_hr = -1
    last_rmssd = -1

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
                    
                    # Read RMSSD
                    if 'rmssd' in data.columns:
                        new_rmssd = data.rmssd.iloc[-1]
                        if new_rmssd != last_rmssd and new_rmssd > 0:
                            state.current_rmssd = new_rmssd
                            last_rmssd = new_rmssd
            else:
                # Generate realistic dummy data for testing
                if count == 1:
                    print_status("output.csv not found - generating dummy data for testing")
                
                # Simulate heart rate based on phase
                if state.phase == "baseline":
                    dummy_hr = 70 + random.uniform(-5, 5)
                    dummy_rmssd = 45 + random.uniform(-10, 15)
                elif state.phase == "warmup":
                    dummy_hr = 85 + random.uniform(-5, 10)
                    dummy_rmssd = 35 + random.uniform(-8, 12)
                elif state.phase in ["exercise1", "exercise2"]:
                    dummy_hr = 140 + random.uniform(-15, 20)
                    dummy_rmssd = 25 + random.uniform(-10, 8)
                elif state.phase == "rest":
                    # Simulate gradual recovery
                    rest_duration = time.time() - (state.rest_start_time or time.time())
                    recovery_factor = min(rest_duration / 60, 1.0)  # Full recovery in 60s
                    baseline_rmssd = state.baseline_rmssd or 45
                    dummy_rmssd = 25 + (baseline_rmssd - 25) * recovery_factor + random.uniform(-5, 5)
                    dummy_hr = 100 - (25 * recovery_factor) + random.uniform(-8, 8)
                else:  # final_rest
                    dummy_hr = 75 + random.uniform(-5, 8)
                    dummy_rmssd = (state.baseline_rmssd or 45) * 0.9 + random.uniform(-5, 10)
                
                state.mean_hr = max(50, dummy_hr)
                state.current_rmssd = max(10, dummy_rmssd)
                
        except Exception as e:
            print_status(f"Data reading error: {e}")

def main():
    """Main function with HRV-optimized exercise protocol"""
    print_header("HRV-OPTIMIZED FITNESS COACH")
    print("🏃‍♂️ Multi-Exercise Heart Rate & HRV Coach")
    print("💚 Featuring adaptive rest periods based on HRV recovery")
    print("\n📋 Session Structure:")
    print("   • 2-minute baseline collection (HR + HRV)")
    print("   • 30-second warm-up")
    print("   • 1-minute exercise set")
    print("   • Adaptive rest period (HRV-optimized)")
    print("   • 1-minute exercise set")
    print("   • 2-minute recovery analysis")
    
    input("\n🚀 Press Enter to start your HRV-optimized session...")

    # Start data collection thread
    data_thread = Thread(target=read_physiological_data, daemon=True)
    data_thread.start()
    print_status("✓ Physiological data monitoring started")

    # Welcome message
    print_header("WELCOME TO YOUR HRV-OPTIMIZED SESSION")
    print_coach_message("Hello! Today I'll use your heart rate variability to optimize your rest periods.")
    print_coach_message("This ensures you get the most effective workout while respecting your body's recovery needs.")
    
    try:
        # Run the enhanced exercise session
        enhanced_exercise_session()
        
    except KeyboardInterrupt:
        print("\n\n👋 Session interrupted by user. Goodbye!")
        state.done = True

    # Cleanup
    print_header("SESSION ENDED")
    print_status("Shutting down HRV monitoring...")
    state.done = True
    
    print_coach_message("🎯 Thank you for trying HRV-optimized training!")
    print_coach_message("💡 Your personalized rest periods help maximize both performance and recovery!")

if __name__ == '__main__':
    main()