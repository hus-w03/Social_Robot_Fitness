#!/usr/bin/env python3
"""
AI Training Periodization Coach with Smart Session Planning
Heart Rate and HRV Monitoring with Intelligent Training Progression
Uses rule-based recovery + AI for training load optimization and session planning
"""

import time
import random
from threading import Thread, Lock
import pandas as pd
import os
import sys
from datetime import datetime, timedelta
import numpy as np
from collections import deque
import pickle
import json

# Machine Learning imports
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, accuracy_score
    ML_AVAILABLE = True
    print("✓ Machine Learning libraries available")
except ImportError:
    ML_AVAILABLE = False
    print("⚠️  sklearn not available - using basic progression only")

# User configuration
AGE = 35
MAX_THRESH = .80
UPPER_THIRD = 0.75
LOWER_THIRD = 0.65
MIN_THRESH = .40

# HRV Safety Thresholds
HRV_GREEN_ZONE = 0.60
HRV_YELLOW_ZONE = 0.30
HRV_RED_ZONE = 0.20
HRV_CRITICAL = 0.15

# Rule-based recovery parameters
MIN_REST_TIME = 15
MAX_REST_TIME = 120
HR_RECOVERY_THRESHOLD = 1.15  # Within 15% of baseline
HRV_RECOVERY_THRESHOLD = 0.7   # Within 70% of baseline

max_hr = 220 - AGE

# Exercise configurations with training load factors
EXERCISE_CONFIG = {
    "squats": {
        "name": "Squats",
        "instructions": "Stand with feet shoulder-width apart, lower your body as if sitting back into a chair, then return to standing.",
        "metronome_normal": 1.2,
        "metronome_fast": 0.8,
        "metronome_slow": 1.8,
        "intensity_modifier": 0.85,
        "hrv_modifier": 1.1,
        "base_duration": 60,
        "training_load_factor": 3.5,  # Relative training stress
        "exercise_type": "strength",
        "muscle_groups": ["quadriceps", "glutes", "core"],
        "difficulty_progression": [45, 60, 75, 90],  # Duration progression in seconds
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
        "base_duration": 60,
        "training_load_factor": 4.0,
        "exercise_type": "strength",
        "muscle_groups": ["quadriceps", "glutes", "hamstrings"],
        "difficulty_progression": [45, 60, 75, 90],
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
        "base_duration": 60,
        "training_load_factor": 6.5,
        "exercise_type": "cardio",
        "muscle_groups": ["cardio", "legs"],
        "difficulty_progression": [30, 45, 60, 75],
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
        "base_duration": 60,
        "training_load_factor": 4.5,
        "exercise_type": "strength",
        "muscle_groups": ["chest", "triceps", "shoulders", "core"],
        "difficulty_progression": [30, 45, 60, 75],
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
        "base_duration": 45,
        "training_load_factor": 8.0,
        "exercise_type": "cardio",
        "muscle_groups": ["full_body"],
        "difficulty_progression": [30, 45, 60, 75],
        "motivational_phrases": [
            "Full body power! You've got this!",
            "Each burpee makes you stronger!",
            "Feel that total body burn!",
            "You're a warrior! Keep fighting!"
        ]
    }
}

class TrainingSessionData:
    """Data structure for storing session information"""
    def __init__(self):
        self.date = datetime.now()
        self.exercises = []
        self.total_duration = 0
        self.training_load = 0
        self.avg_hr = 0
        self.peak_hr = 0
        self.baseline_hr = 0
        self.baseline_hrv = 0
        self.avg_hrv = 0
        self.recovery_time = 0
        self.user_perceived_exertion = 0  # 1-10 scale
        self.user_enjoyment = 0  # 1-10 scale
        self.completion_rate = 0  # percentage completed
        self.performance_score = 0  # derived metric
        self.fatigue_score = 0  # derived metric

class TrainingPeriodizationAI:
    """AI system for intelligent training session planning and periodization"""
    
    def __init__(self):
        self.session_history = deque(maxlen=200)  # Store last 200 sessions
        self.performance_model = None
        self.load_recommendation_model = None
        self.exercise_selection_model = None
        self.scaler = StandardScaler() if ML_AVAILABLE else None
        self.model_trained = False
        self.user_preferences = {}
        self.training_parameters = {
            'fitness_level': 1,  # 1-5 scale
            'weekly_frequency': 3,
            'preferred_exercises': [],
            'goals': [],  # strength, cardio, general_fitness
            'available_time': 60,  # minutes per session
        }
    
    def calculate_training_load(self, session_data):
        """Calculate session training load based on multiple factors"""
        base_load = 0
        
        for exercise in session_data.exercises:
            exercise_config = EXERCISE_CONFIG.get(exercise['name'], {})
            duration_factor = exercise['duration'] / exercise_config.get('base_duration', 60)
            load_factor = exercise_config.get('training_load_factor', 5.0)
            intensity_factor = exercise.get('intensity_multiplier', 1.0)
            
            exercise_load = load_factor * duration_factor * intensity_factor
            base_load += exercise_load
        
        # Adjust for physiological response
        hr_stress = (session_data.avg_hr / session_data.baseline_hr - 1) * 10 if session_data.baseline_hr > 0 else 0
        hrv_stress = (1 - session_data.avg_hrv / session_data.baseline_hrv) * 5 if session_data.baseline_hrv > 0 else 0
        
        total_load = base_load * (1 + hr_stress * 0.1 + hrv_stress * 0.1)
        return max(total_load, 0)
    
    def calculate_performance_score(self, session_data):
        """Calculate session performance score"""
        completion_score = session_data.completion_rate / 100.0
        efficiency_score = min(session_data.avg_hr / (session_data.baseline_hr * 1.3), 1.0) if session_data.baseline_hr > 0 else 0.5
        hrv_score = min(session_data.avg_hrv / session_data.baseline_hrv, 1.0) if session_data.baseline_hrv > 0 else 0.5
        
        performance = (completion_score * 0.4 + efficiency_score * 0.3 + hrv_score * 0.3) * 10
        return max(min(performance, 10), 0)
    
    def calculate_fatigue_score(self, days_back=7):
        """Calculate cumulative fatigue over recent days"""
        if len(self.session_history) == 0:
            return 0
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        recent_sessions = [s for s in self.session_history if s.date > cutoff_date]
        
        if not recent_sessions:
            return 0
        
        total_load = sum(s.training_load for s in recent_sessions)
        avg_recovery = np.mean([s.recovery_time for s in recent_sessions])
        avg_hrv_ratio = np.mean([s.avg_hrv / s.baseline_hrv for s in recent_sessions if s.baseline_hrv > 0])
        
        # Normalize and combine factors
        load_factor = min(total_load / (len(recent_sessions) * 20), 2.0)  # Normalize by expected load
        recovery_factor = min(avg_recovery / 60, 2.0)  # Normalize by expected recovery time
        hrv_factor = max(2 - avg_hrv_ratio, 0) if avg_hrv_ratio > 0 else 1.0
        
        fatigue = (load_factor + recovery_factor + hrv_factor) / 3 * 10
        return max(min(fatigue, 10), 0)
    
    def get_session_features(self, target_date=None):
        """Extract features for ML models"""
        if not target_date:
            target_date = datetime.now()
        
        # Historical performance features
        recent_sessions = [s for s in self.session_history if s.date > target_date - timedelta(days=30)]
        if not recent_sessions:
            return np.zeros(15)  # Return zero features if no history
        
        features = [
            len(recent_sessions),  # Session frequency
            np.mean([s.performance_score for s in recent_sessions]),  # Avg performance
            np.mean([s.training_load for s in recent_sessions]),  # Avg training load
            np.mean([s.user_perceived_exertion for s in recent_sessions]),  # Avg RPE
            np.mean([s.recovery_time for s in recent_sessions]),  # Avg recovery time
            self.calculate_fatigue_score(7),  # 7-day fatigue
            self.calculate_fatigue_score(14),  # 14-day fatigue
            (target_date - recent_sessions[-1].date).days if recent_sessions else 7,  # Days since last session
            target_date.weekday(),  # Day of week
            len([s for s in recent_sessions if s.date > target_date - timedelta(days=7)]),  # Weekly frequency
            self.training_parameters['fitness_level'],  # User fitness level
            self.training_parameters['available_time'],  # Available time
            np.mean([s.user_enjoyment for s in recent_sessions]),  # User enjoyment
            np.std([s.performance_score for s in recent_sessions]) if len(recent_sessions) > 1 else 0,  # Performance variability
            max([s.training_load for s in recent_sessions]) if recent_sessions else 0  # Peak recent load
        ]
        
        return np.array(features)
    
    def recommend_training_load(self, target_date=None):
        """Recommend optimal training load for next session"""
        if not ML_AVAILABLE or not self.model_trained or len(self.session_history) < 10:
            return self._rule_based_load_recommendation()
        
        features = self.get_session_features(target_date)
        if np.all(features == 0):
            return self._rule_based_load_recommendation()
        
        try:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            recommended_load = self.load_recommendation_model.predict(features_scaled)[0]
            return max(min(recommended_load, 50), 5)  # Clamp between 5-50
        except Exception as e:
            print(f"ML load recommendation failed: {e}")
            return self._rule_based_load_recommendation()
    
    def _rule_based_load_recommendation(self):
        """Fallback rule-based load recommendation"""
        fatigue = self.calculate_fatigue_score()
        base_load = 15  # Base training load
        
        if fatigue > 7:
            return base_load * 0.6  # Reduce load if high fatigue
        elif fatigue < 3:
            return base_load * 1.3  # Increase load if low fatigue
        else:
            return base_load
    
    def recommend_exercises(self, target_load, available_time=60):
        """Recommend optimal exercise selection and progression"""
        fatigue = self.calculate_fatigue_score()
        recent_exercises = []
        
        # Get recent exercise history
        if len(self.session_history) > 0:
            recent_sessions = list(self.session_history)[-5:]  # Last 5 sessions
            for session in recent_sessions:
                recent_exercises.extend([ex['name'] for ex in session.exercises])
        
        # Exercise selection strategy
        available_exercises = list(EXERCISE_CONFIG.keys())
        recommended_exercises = []
        
        # Avoid recently performed exercises for variety
        exercise_counts = {ex: recent_exercises.count(ex) for ex in available_exercises}
        sorted_exercises = sorted(available_exercises, key=lambda x: (exercise_counts[x], random.random()))
        
        # Select exercises based on training load target and fatigue
        current_load = 0
        max_exercises = 3 if fatigue > 6 else 4 if fatigue > 3 else 5
        
        for exercise in sorted_exercises:
            if len(recommended_exercises) >= max_exercises:
                break
                
            config = EXERCISE_CONFIG[exercise]
            exercise_load = config['training_load_factor']
            
            # Adjust duration based on target load and fatigue
            if fatigue > 6:
                duration = min(config['difficulty_progression'])  # Easiest
            elif fatigue < 3:
                duration = max(config['difficulty_progression'])  # Hardest
            else:
                duration = config['base_duration']  # Standard
            
            # Check if adding this exercise would exceed target load
            estimated_load = exercise_load * (duration / config['base_duration'])
            if current_load + estimated_load <= target_load * 1.2:  # Allow 20% tolerance
                recommended_exercises.append({
                    'name': exercise,
                    'duration': duration,
                    'intensity_multiplier': 1.0,
                    'estimated_load': estimated_load
                })
                current_load += estimated_load
        
        return recommended_exercises
    
    def predict_performance(self, planned_exercises, target_date=None):
        """Predict expected performance for planned session"""
        if not ML_AVAILABLE or not self.model_trained:
            return self._rule_based_performance_prediction(planned_exercises)
        
        # This would use the performance model to predict outcomes
        return 7.5  # Placeholder
    
    def _rule_based_performance_prediction(self, planned_exercises):
        """Rule-based performance prediction"""
        fatigue = self.calculate_fatigue_score()
        total_load = sum(ex.get('estimated_load', 5) for ex in planned_exercises)
        
        # Simple heuristic: performance decreases with fatigue and very high loads
        base_performance = 8.0
        fatigue_penalty = fatigue * 0.3
        load_penalty = max(0, (total_load - 20) * 0.1)
        
        predicted_performance = base_performance - fatigue_penalty - load_penalty
        return max(min(predicted_performance, 10), 1)
    
    def record_session(self, session_data):
        """Record completed session data"""
        session_data.training_load = self.calculate_training_load(session_data)
        session_data.performance_score = self.calculate_performance_score(session_data)
        session_data.fatigue_score = self.calculate_fatigue_score()
        
        self.session_history.append(session_data)
        
        # Retrain models periodically
        if len(self.session_history) % 10 == 0 and len(self.session_history) >= 20:
            self.train_models()
    
    def train_models(self):
        """Train ML models with session history"""
        if not ML_AVAILABLE or len(self.session_history) < 20:
            return False
        
        try:
            # Prepare training data
            features = []
            load_targets = []
            performance_targets = []
            
            for i, session in enumerate(list(self.session_history)[10:]):  # Skip first 10 for features
                session_features = self.get_session_features(session.date)
                features.append(session_features)
                load_targets.append(session.training_load)
                performance_targets.append(session.performance_score)
            
            if len(features) < 10:
                return False
            
            X = np.array(features)
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            
            # Train load recommendation model
            self.load_recommendation_model = RandomForestRegressor(n_estimators=50, random_state=42)
            self.load_recommendation_model.fit(X_scaled, load_targets)
            
            # Train performance prediction model
            self.performance_model = RandomForestRegressor(n_estimators=50, random_state=42)
            self.performance_model.fit(X_scaled, performance_targets)
            
            self.model_trained = True
            print(f"✓ AI models trained with {len(features)} sessions")
            return True
            
        except Exception as e:
            print(f"❌ Failed to train AI models: {e}")
            return False
    
    def get_training_recommendations(self, available_time=60):
        """Get comprehensive training recommendations for next session"""
        recommended_load = self.recommend_training_load()
        recommended_exercises = self.recommend_exercises(recommended_load, available_time)
        predicted_performance = self.predict_performance(recommended_exercises)
        fatigue = self.calculate_fatigue_score()
        
        # Generate insights and recommendations
        insights = []
        if fatigue > 7:
            insights.append("High fatigue detected - recommend lighter session or rest day")
        elif fatigue < 3:
            insights.append("Low fatigue - good opportunity to increase training intensity")
        
        if len(self.session_history) > 5:
            recent_performance = np.mean([s.performance_score for s in list(self.session_history)[-5:]])
            if recent_performance > 8:
                insights.append("Strong recent performance - consider progressive overload")
            elif recent_performance < 6:
                insights.append("Performance declining - may need recovery focus")
        
        return {
            'recommended_load': recommended_load,
            'recommended_exercises': recommended_exercises,
            'predicted_performance': predicted_performance,
            'current_fatigue': fatigue,
            'insights': insights,
            'session_duration': sum(ex['duration'] for ex in recommended_exercises) + 10  # +10 for rest
        }
    
    def save_models(self, filename="training_ai_models.pkl"):
        """Save trained models and data"""
        try:
            model_data = {
                'load_recommendation_model': self.load_recommendation_model,
                'performance_model': self.performance_model,
                'scaler': self.scaler,
                'session_history': list(self.session_history),
                'training_parameters': self.training_parameters,
                'model_trained': self.model_trained
            }
            with open(filename, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"✓ AI models saved to {filename}")
        except Exception as e:
            print(f"❌ Failed to save models: {e}")
    
    def load_models(self, filename="training_ai_models.pkl"):
        """Load previously trained models"""
        if not ML_AVAILABLE:
            return False
        
        try:
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.load_recommendation_model = model_data.get('load_recommendation_model')
                self.performance_model = model_data.get('performance_model')
                self.scaler = model_data.get('scaler')
                self.session_history = deque(model_data.get('session_history', []), maxlen=200)
                self.training_parameters = model_data.get('training_parameters', self.training_parameters)
                self.model_trained = model_data.get('model_trained', False)
                
                print(f"✓ AI models loaded from {filename}")
                print(f"  Session history: {len(self.session_history)} sessions")
                print(f"  Models trained: {self.model_trained}")
                return True
        except Exception as e:
            print(f"❌ Failed to load models: {e}")
        
        return False

# Rule-based recovery monitoring (keeping this simple and reliable)
def rule_based_recovery_monitor(hr_data, hrv_data, baseline_hr, baseline_hrv, time_elapsed):
    """Rule-based recovery assessment"""
    if len(hr_data) < 5 or time_elapsed < MIN_REST_TIME:
        return False
    
    current_hr = hr_data[-1]
    recent_hr = hr_data[-5:]
    
    # HR recovery criteria
    hr_threshold = baseline_hr * HR_RECOVERY_THRESHOLD
    hr_stable = abs(max(recent_hr) - min(recent_hr)) < 5
    hr_declining = np.polyfit(range(len(recent_hr)), recent_hr, 1)[0] <= 0
    
    # HRV recovery criteria
    hrv_recovered = True
    if len(hrv_data) > 0 and hrv_data[-1] > 0 and baseline_hrv > 0:
        current_hrv = hrv_data[-1]
        hrv_threshold = baseline_hrv * HRV_RECOVERY_THRESHOLD
        hrv_recovered = current_hrv >= hrv_threshold
    
    # Time limit
    if time_elapsed >= MAX_REST_TIME:
        return True
    
    # Recovery criteria
    recovery_criteria = [
        current_hr <= hr_threshold,
        hr_stable,
        hr_declining,
        hrv_recovered
    ]
    
    return sum(recovery_criteria) >= 3  # Need at least 3 out of 4 criteria

# Global state and AI system
class ExerciseState:
    def __init__(self):
        self.lock = Lock()
        self._rest_hr = None
        self._mean_hr = -1
        self._rest_hrv = None
        self._current_hrv = -1
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
        
        # Recovery monitoring
        self.in_recovery = False
        self.recovery_start_time = None
        self.recovery_hr_data = []
        self.recovery_hrv_data = []
        
        # Session tracking
        self.current_session = TrainingSessionData()
        self.session_start_time = None
        self.hr_data_session = []
        self.hrv_data_session = []
        
    # Properties (same as before)
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

state = ExerciseState()
training_ai = TrainingPeriodizationAI()

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def print_coach_message(message):
    """Print coach message with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] 🤖 COACH: {message}")

def print_status(message):
    """Print status message"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] ℹ️  STATUS: {message}")

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
            print("\n\n👋 Session interrupted by user. Saving progress...")
            training_ai.save_models()
            state.done = True
            sys.exit(0)

def display_ai_recommendations():
    """Display AI-generated training recommendations"""
    print_header("AI TRAINING RECOMMENDATIONS")
    
    recommendations = training_ai.get_training_recommendations()
    
    print_coach_message("Based on your training history and current status, here are my AI recommendations:")
    print(f"🎯 Recommended Training Load: {recommendations['recommended_load']:.1f}")
    print(f"⏱️  Estimated Session Duration: {recommendations['session_duration']} minutes")
    print(f"📊 Predicted Performance: {recommendations['predicted_performance']:.1f}/10")
    print(f"😴 Current Fatigue Level: {recommendations['current_fatigue']:.1f}/10")
    
    print("\n🏋️‍♂️ Recommended Exercises:")
    for i, exercise in enumerate(recommendations['recommended_exercises'], 1):
        exercise_name = EXERCISE_CONFIG[exercise['name']]['name']
        print(f"  {i}. {exercise_name} - {exercise['duration']} seconds (Load: {exercise['estimated_load']:.1f})")
    
    if recommendations['insights']:
        print("\n💡 AI Insights:")
        for insight in recommendations['insights']:
            print(f"  • {insight}")
    
    # Give user choice
    choice = get_user_input("Would you like to (1) follow AI recommendations, (2) customize session, or (3) view training history?", ['1', '2', '3'])
    
    if choice == '1':
        return recommendations['recommended_exercises']
    elif choice == '2':
        return customize_session()
    else:
        display_training_history()
        return display_ai_recommendations()  # Recursive call after viewing history

def customize_session():
    """Allow user to customize their session"""
    print_header("SESSION CUSTOMIZATION")
    
    available_exercises = list(EXERCISE_CONFIG.keys())
    selected_exercises = []
    
    print_coach_message("Let's customize your session. Choose exercises and durations:")
    
    for i, exercise_key in enumerate(available_exercises, 1):
        exercise_name = EXERCISE_CONFIG[exercise_key]['name']
        config = EXERCISE_CONFIG[exercise_key]
        print(f"{i}. {exercise_name} (Base: {config['base_duration']}s, Load Factor: {config['training_load_factor']})")
    
    while len(selected_exercises) < 5:  # Max 5 exercises
        try:
            choice = get_user_input(f"Select exercise {len(selected_exercises)+1} (1-{len(available_exercises)}, or 'done' to finish)")
            
            if choice.lower() == 'done':
                break
            
            exercise_idx = int(choice) - 1
            if 0 <= exercise_idx < len(available_exercises):
                exercise_key = available_exercises[exercise_idx]
                config = EXERCISE_CONFIG[exercise_key]
                
                # Get duration
                duration = int(get_user_input(f"Duration for {config['name']} (seconds, recommended: {config['base_duration']})") or config['base_duration'])
                
                selected_exercises.append({
                    'name': exercise_key,
                    'duration': duration,
                    'intensity_multiplier': 1.0,
                    'estimated_load': config['training_load_factor'] * (duration / config['base_duration'])
                })
                
                print(f"✓ Added {config['name']} for {duration} seconds")
            else:
                print("❌ Invalid exercise number")
                
        except ValueError:
            if choice.lower() != 'done':
                print("❌ Please enter a valid number or 'done'")
    
    return selected_exercises

def display_training_history():
    """Display training history and analytics"""
    print_header("TRAINING HISTORY & ANALYTICS")
    
    if len(training_ai.session_history) == 0:
        print_coach_message("No training history available yet. Complete your first session to see analytics!")
        return
    
    sessions = list(training_ai.session_history)
    recent_sessions = sessions[-10:]  # Last 10 sessions
    
    print_coach_message("Your Training Analytics:")
    print(f"📈 Total Sessions: {len(sessions)}")
    print(f"📊 Average Performance: {np.mean([s.performance_score for s in sessions]):.1f}/10")
    print(f"🏋️‍♂️ Average Training Load: {np.mean([s.training_load for s in sessions]):.1f}")
    print(f"⚡ Current Fatigue: {training_ai.calculate_fatigue_score():.1f}/10")
    
    if len(sessions) >= 5:
        recent_performance = np.mean([s.performance_score for s in recent_sessions])
        older_performance = np.mean([s.performance_score for s in sessions[:-10]]) if len(sessions) > 10 else recent_performance
        
        trend = "improving" if recent_performance > older_performance else "declining" if recent_performance < older_performance else "stable"
        print(f"📈 Performance Trend: {trend}")
    
    print("\n🏃‍♂️ Recent Sessions:")
    for session in recent_sessions[-5:]:
        print(f"  {session.date.strftime('%Y-%m-%d')}: Performance {session.performance_score:.1f}, Load {session.training_load:.1f}")
    
    input("\n⏸️  Press Enter to continue...")

def start_session_tracking():
    """Initialize session tracking"""
    state.current_session = TrainingSessionData()
    state.session_start_time = time.time()
    state.hr_data_session = []
    state.hrv_data_session = []

def finalize_session():
    """Complete session tracking and analysis"""
    print_header("SESSION COMPLETE - COLLECTING FEEDBACK")
    
    # Calculate session statistics
    if state.hr_data_session:
        state.current_session.avg_hr = np.mean(state.hr_data_session)
        state.current_session.peak_hr = max(state.hr_data_session)
    
    if state.hrv_data_session:
        state.current_session.avg_hrv = np.mean([h for h in state.hrv_data_session if h > 0])
    
    state.current_session.baseline_hr = state.rest_hr or 70
    state.current_session.baseline_hrv = state.rest_hrv or 30
    state.current_session.total_duration = (time.time() - state.session_start_time) / 60  # minutes
    
    # Get user feedback
    try:
        rpe = int(get_user_input("Rate your perceived exertion (1-10, where 10 is maximum effort)") or "5")
        enjoyment = int(get_user_input("How much did you enjoy this session? (1-10, where 10 is very enjoyable)") or "7")
        
        state.current_session.user_perceived_exertion = max(1, min(10, rpe))
        state.current_session.user_enjoyment = max(1, min(10, enjoyment))
        state.current_session.completion_rate = 100  # Assume completed if we reach this point
        
    except ValueError:
        # Default values if user input is invalid
        state.current_session.user_perceived_exertion = 5
        state.current_session.user_enjoyment = 7
        state.current_session.completion_rate = 100
    
    # Record session in AI system
    training_ai.record_session(state.current_session)
    
    # Display session summary
    print_coach_message("Session Summary:")
    print(f"📊 Performance Score: {state.current_session.performance_score:.1f}/10")
    print(f"🏋️‍♂️ Training Load: {state.current_session.training_load:.1f}")
    print(f"⏱️  Duration: {state.current_session.total_duration:.1f} minutes")
    print(f"❤️  Average HR: {state.current_session.avg_hr:.0f} BPM")
    
    # AI insights for next session
    next_recommendations = training_ai.get_training_recommendations()
    print(f"\n🔮 AI Prediction for Next Session:")
    print(f"   Recommended rest before next session: {1 if next_recommendations['current_fatigue'] < 4 else 2 if next_recommendations['current_fatigue'] < 7 else 3} days")

def main():
    """Main function"""
    print_header("AI TRAINING PERIODIZATION COACH")
    print("🧠 Intelligent Training Session Planning & Load Management")
    print("📈 Machine Learning-Enhanced Training Progression")
    print("🎯 Personalized Exercise Programming")
    
    # Load previous AI models
    training_ai.load_models()
    
    input("\n🚀 Press Enter to start your AI-enhanced training session...")
    
    # Initialize session tracking
    start_session_tracking()
    
    # Get AI recommendations for today's session
    recommended_exercises = display_ai_recommendations()
    
    if not recommended_exercises:
        print_coach_message("No exercises selected. Ending session.")
        return
    
    print_coach_message(f"Great! We'll be doing {len(recommended_exercises)} exercises today.")
    
    # Execute session (simplified for demo)
    for i, exercise in enumerate(recommended_exercises, 1):
        exercise_name = EXERCISE_CONFIG[exercise['name']]['name']
        duration = exercise['duration']
        
        print_header(f"EXERCISE {i}/{len(recommended_exercises)}: {exercise_name.upper()}")
        
        # Record exercise in session
        state.current_session.exercises.append(exercise)
        
        print_coach_message(f"Starting {exercise_name} for {duration} seconds...")
        
        # Simulate exercise execution (in real implementation, this would be the full exercise loop)
        for remaining in range(duration, 0, -5):
            print_status(f"{exercise_name}: {remaining} seconds remaining")
            time.sleep(1)  # Shortened for demo
            
            # Simulate data collection
            dummy_hr = 70 + random.uniform(20, 50)
            dummy_hrv = 30 + random.uniform(-10, 10)
            state.hr_data_session.append(dummy_hr)
            state.hrv_data_session.append(dummy_hrv)
        
        print_coach_message(f"✅ {exercise_name} complete!")
        
        # Rest period between exercises (rule-based)
        if i < len(recommended_exercises):
            print_status("Starting recovery monitoring...")
            rest_start = time.time()
            hr_recovery = []
            hrv_recovery = []
            
            while True:
                elapsed = time.time() - rest_start
                
                # Simulate recovery data
                recovery_hr = max(70, dummy_hr - (elapsed * 2))
                recovery_hrv = min(35, dummy_hrv + (elapsed * 0.5))
                hr_recovery.append(recovery_hr)
                hrv_recovery.append(recovery_hrv)
                
                if rule_based_recovery_monitor(hr_recovery, hrv_recovery, state.rest_hr or 70, state.rest_hrv or 30, elapsed):
                    print_coach_message(f"✅ Recovery complete after {elapsed:.0f} seconds!")
                    break
                
                print_status(f"Recovery: {elapsed:.0f}s | HR: {recovery_hr:.0f} BPM")
                time.sleep(2)
    
    # Complete session
    finalize_session()
    
    # Save models
    training_ai.save_models()
    
    print_header("TRAINING SESSION COMPLETE")
    print_coach_message("Thank you for training with me! Your data has been recorded for future session planning.")

if __name__ == '__main__':
    main()