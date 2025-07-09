# Socially Assistive Robot for Cardio Exercises

A comprehensive system that combines social robotics with physiological monitoring to provide personalized, adaptive cardio exercise coaching

## Features

- **Multi-Exercise Support:** 5 evidence-based cardio exercises suitable for various fitness levels
- **Real-time Physiological Monitoring:** Heart rate and HRV analysis using Polar chest strap sensors
- **Adaptive Feedback:** Intelligent intensity adjustment based on physiological responses
- **Social Robot Integration:** QT Robot with speech, gestures and emotional expressions
- **Personalized Profiles:** User-specific configurations and progress tracking

## Supported Exercises
1. **Squats**
2. **Lunges**
3. **High Knees**
4. **Push-ups**
5. **Burpees**


## System Requirements
- Python 3.8+
- ROS (Robot Operating System)
- QT Robot with ROSbridge
- Polar H10 chest strap sensor
- Linux environment (recommended)


## Installation

1. Clone the repository:
```bash
git clone <https://github.com/hus-w03/Social_Robot_Fitness.git>
cd Social_Robot_Fitness-main/Social_robot
```

2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate #On windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip3 install bleak hrvanalysis
```

4. Start sympathetic_plot.py and bt_analysis.py
```bash
python3 -W ignore sympathetic_plot.py & python3 bt_analysis.py
```

5. In another terminal, start coaching_feedback.py
```bash
python3 coaching_feedback.py
```
