import obd
import time

# Establish an OBD-II connection
connection = obd.OBD()

# Check if the connection is successful
if connection.is_connected():
    print("OBD-II connection successful")
else:
    print("OBD-II connection failed. Make sure your OBD-II adapter is connected properly.")
    exit()

# Function to get and print the speed
def get_speed():
    cmd = obd.commands.SPEED  # OBD-II command for speed
    response = connection.query(cmd)

    if response.is_null():
        print("Speed data not available.")
    else:
        speed_kph = response.value.magnitude  # Speed in kilometers per hour
        speed_mph = obd.Unit.kph2mph(speed_kph)  # Convert speed to miles per hour
        print(f"Speed: {speed_kph} km/h | {speed_mph} mph")

# Main loop to continuously get and print speed
try:
    while True:
        get_speed()
        time.sleep(1)  # Update speed every second

except KeyboardInterrupt:
    print("Script terminated by user.")
    connection.close()