from bleak import BleakClient
from dataclasses import dataclass
from support_functions_fixed import ewma_vectorized
from math import ceil
from queue import Queue

import asyncio
import hrvanalysis as hrv
import os
from scipy.signal import find_peaks
from sensor_config import *
import signal
import sys
import time

import rospy
from std_msgs.msg import String


ADDRESS = h10_880
DEBUG = False
ROLLING_WINDOW = True
WINDOW_PERIOD = 120 # 120 - jumping jacks; 60 - BreATHING AND PSYCH

if DEBUG:
    import random


@dataclass
class HeartRateData:
    time_stamp: float
    rr_interval: int


global cur_hr_data
global hr_data_q
global data_out
global initial_time
global sensor_connected
global resp_pub

# Keyboard Interrupt Handler
def keyboard_interrupt_handler(signum, frame):
    print("----------------Recording stopped------------------------")
    write_to_file()
    sys.exit()


# reads bluetooth data and puts HeartRateData object into hr_data queue
async def data_conv(sender, data):
    global hr_data_q
    byte0 = data[0]
    uint8_format = (byte0 & 1) == 0
    energy_expenditure = ((byte0 >> 3) & 1) == 1
    rr_interval = ((byte0 >> 4) & 1) == 1

    if not rr_interval:
        return

    first_rr_byte = 2
    if uint8_format:
        # hr = data[1]
        pass
    else:
        # hr = (data[2] << 8) | data[1] # uint16
        first_rr_byte += 1
    if energy_expenditure:
        # ee = (data[first_rr_byte + 1] << 8) | data[first_rr_byte]
        first_rr_byte += 2

    # time is now calculated and saved closer to when the data was received
    now = time.time()
    readings = []
    
    # FIX: Check bounds before accessing data[i + 1]
    for i in range(first_rr_byte, len(data) - 1, 2):  # Changed to len(data) - 1
        if i + 1 < len(data):  # Additional safety check
            ibi = (data[i + 1] << 8) | data[i]
            ibi = ceil(ibi / 1024 * 1000)
            # assume ibi is the full int of rr interval
            readings.append(ibi)

    for i in range(len(readings)):
        # fake the time stamp so curve is smooth, assumes data is produced at a consistent rate
        hr_data_q.put(HeartRateData(float(now + 1.0 * ((i + 1) / len(readings))), readings[i]))


async def buffer():
    global hr_data_q, cur_hr_data, data_out, initial_time

    sensor_data = hr_data_q.get()
    cur_hr_data.append(sensor_data.rr_interval)

    # full hr data list can be accessed with data_out['rr_interval']
    data_out['time'].append(int(sensor_data.time_stamp - initial_time))
    data_out['rr_interval'].append(sensor_data.rr_interval)
    if len(cur_hr_data) > 2:
        data_out['hr'].append(hrv.get_time_domain_features(cur_hr_data[-2:])['mean_hr'])
    else:
        data_out['hr'].append(-1)
    
    # this doesn't give exactly 2 minutes of data, but should be within ~1 second accuracy
    if sensor_data.time_stamp - initial_time > WINDOW_PERIOD:
        filtered_data = ewma_vectorized(cur_hr_data, 0.5)

        time_features = hrv.get_time_domain_features(filtered_data)
        freq_features = hrv.get_frequency_domain_features(filtered_data)
        breathing_rate = len(list(find_peaks(filtered_data)[0])) * 60 / WINDOW_PERIOD

        # UPDATE TO 1 MINUTE INTERVAL, 30 SECONDS IF POSSIBLE AND DATA LOOKS ACCURATE

        data_out['mean_hr'].append(time_features['mean_hr'])
        data_out['max_hr'].append(time_features['max_hr'])
        data_out['min_hr'].append(time_features['min_hr'])
        data_out['sdnn'].append(time_features['sdnn'])
        data_out['rmssd'].append(time_features['rmssd'])

        data_out['lf'].append(freq_features['lf'])
        data_out['hf'].append(freq_features['hf'])
        data_out['lf_hf_ratio'].append(freq_features['lf_hf_ratio'])

        data_out['breath_rate'].append(breathing_rate)
        if ROLLING_WINDOW:
            cur_hr_data.pop(0)
        else:
            cur_hr_data.clear()
            initial_time = time.time()
    else:
        data_out['mean_hr'].append(-1)
        data_out['max_hr'].append(-1)
        data_out['min_hr'].append(-1)
        data_out['sdnn'].append(-1)
        data_out['rmssd'].append(-1)

        data_out['lf'].append(-1)
        data_out['hf'].append(-1)
        data_out['lf_hf_ratio'].append(-1)

        data_out['breath_rate'].append(-1)


# Asynchronous task to start the data stream
async def run(client, debug=False):
    global hr_data_q, initial_time, sensor_connected
    sensor_connected = 0
    # Writing characteristic description to control point for request of UUID (defined above) ##
    if not debug:
        # FIX: Changed from await client.is_connected() to client.is_connected
        if client.is_connected:
            print("---------Device connected--------------")

            model_number = await client.read_gatt_char(MODEL_NBR_UUID)
            print("Model Number: {0}".format("".join(map(chr, model_number))))

            manufacturer_name = await client.read_gatt_char(MANUFACTURER_NAME_UUID)
            print("Manufacturer Name: {0}".format("".join(map(chr, manufacturer_name))))

            battery_level = await client.read_gatt_char(BATTERY_LEVEL_UUID)
            print("Battery Level: {0}%".format(int(battery_level[0])))

            await client.start_notify(HR_UUID, data_conv)
    # delay to let data settle
    await asyncio.sleep(15)
    hr_data_q.queue.clear()
    print("Collecting RR data")
    initial_time = time.time()
    sensor_connected = 1

    while not rospy.is_shutdown():
        await asyncio.sleep(0.5)
        while not hr_data_q.empty():
            await buffer()


async def write_loop():
    while not rospy.is_shutdown():
        await asyncio.sleep(1)  # updates the file every 1 second
        write_to_file()


def write_to_file():
    global data_out

    try:
        # FIX: Check that all lists have the same length
        if not data_out:
            return
        
        # Get the minimum length to avoid index errors
        min_length = min(len(data_out[key]) for key in data_out.keys())
        
        if min_length == 0:
            return
        
        headers = ""
        for key in data_out.keys():
            headers += (str(key) + ',')
        headers = headers.rstrip(',') + '\n'  # Remove trailing comma

        data = ""
        for i in range(min_length):  # Use min_length instead of len(data_out['time'])
            row = ""
            for key in data_out.keys():
                if i < len(data_out[key]):  # Additional safety check
                    row += str(data_out[key][i]) + ','
                else:
                    row += "-1,"  # Default value if index doesn't exist
            data += row.rstrip(',') + '\n'  # Remove trailing comma

        out = headers + data
        with open("output.csv", "w") as f:  # Use context manager
            f.write(out)

    except Exception as e:
        print(f"Error writing to file: {e}")


async def produce_debug_data():
    global hr_data_q
    while not rospy.is_shutdown():
        await asyncio.sleep(0.5)
        print(time.time())
        hr_data_q.put(HeartRateData(time.time(), random.randrange(600, 1000)))


def sensor_callback(msg):
    """ROS callback function for sensor connection requests"""
    global resp_pub, sensor_connected
    
    print(f"Received sensor connection request: {msg.data}")
    
    # Wait for sensor to be connected
    while not sensor_connected and not rospy.is_shutdown():
        time.sleep(0.5)
    
    if sensor_connected and resp_pub is not None:
        # Publish response that sensor is connected
        response_msg = String()
        response_msg.data = "sensor_connected"
        resp_pub.publish(response_msg)
        print("Published sensor_connected response")


async def ros_spin_async():
    """Async wrapper for ROS spinning"""
    while not rospy.is_shutdown():
        await asyncio.sleep(0.1)  # Small delay to prevent busy waiting


async def main():
    global hr_data_q, data_out, cur_hr_data, resp_pub
    
    # Initialize ROS node
    try:
        rospy.init_node('bt_heart_rate_sensor', anonymous=True)
        print("✓ ROS node initialized")
    except Exception as e:
        print(f"✗ Failed to initialize ROS node: {e}")
        print("Make sure roscore is running")
        return
    
    # Initialize ROS publisher and subscriber
    try:
        resp_pub = rospy.Publisher('/ros_resp', String, queue_size=10)
        sensor_sub = rospy.Subscriber('/sensor_connected', String, sensor_callback)
        print("✓ ROS publisher and subscriber initialized")
        
        # Wait for connections to establish
        time.sleep(2)
        
    except Exception as e:
        print(f"✗ Failed to initialize ROS communication: {e}")
        return
    
    # Initialize data structures
    cur_hr_data = []
    hr_data_q = Queue(maxsize=120)

    data_out = {'time': [],
                'rr_interval': [],
                'lf': [],
                'hf': [],
                'lf_hf_ratio': [],
                'hr': [],
                'mean_hr': [],
                'max_hr': [],
                'min_hr': [],
                'rmssd': [],
                'sdnn': [],
                'breath_rate': []
                }

    try:
        async with BleakClient(ADDRESS) as client:
            signal.signal(signal.SIGINT, keyboard_interrupt_handler)
            tasks = [
                asyncio.ensure_future(run(client, DEBUG)),
                asyncio.ensure_future(write_loop()),
                asyncio.ensure_future(ros_spin_async()),  # Add ROS spinning task
            ]
            if DEBUG:
                tasks.append(asyncio.ensure_future(produce_debug_data()))

            await asyncio.gather(*tasks)
    except Exception as e:
        print(f"Error in main execution: {e}")
    finally:
        keyboard_interrupt_handler(None, None)


if __name__ == "__main__":
    os.environ["PYTHONASYNCIODEBUG"] = str(1)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())