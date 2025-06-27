from bleak.uuids import uuid16_dict

uuid16_dict = {v: k for k, v in uuid16_dict.items()}

# This is the device MAC ID
h10_593 = "CC:9D:24:5B:80:8C"
h10_770 = "EE:A0:83:B7:1E:06"
h10_880 = "24:AC:AC:01:12:91"

# UUID for model number
MODEL_NBR_UUID = "0000{0:x}-0000-1000-8000-00805f9b34fb".format(
    uuid16_dict.get("Model Number String")
)

# UUID for manufacturer name
MANUFACTURER_NAME_UUID = "0000{0:x}-0000-1000-8000-00805f9b34fb".format(
    uuid16_dict.get("Manufacturer Name String")
)

# UUID for battery level
BATTERY_LEVEL_UUID = "0000{0:x}-0000-1000-8000-00805f9b34fb".format(
    uuid16_dict.get("Battery Level")
)

# UUID for connection establishment with device
PMD_SERVICE = "FB005C80-02E7-F387-1CAD-8ACD2D8DF0C8"

# UUID for Request of stream settings
PMD_CONTROL = "FB005C81-02E7-F387-1CAD-8ACD2D8DF0C8"

# UUID for Request of start stream
PMD_DATA = "FB005C82-02E7-F387-1CAD-8ACD2D8DF0C8"

# UUID for Request of HR stream
HR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"
