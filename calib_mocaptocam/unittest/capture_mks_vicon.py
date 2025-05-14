import socket
import os
import csv
from datetime import datetime
from pynput import keyboard

# UDP Configuration
ip = "172.20.183.220"
port = 5005

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((ip, port))

# CSV Setup
output_dir = "/root/workspace/ros_ws/src/rt-cosmik/output"
os.makedirs(output_dir, exist_ok=True)
filename = os.path.join(output_dir, "mks_pose.csv")
columns = ["time", "marker_data"]

# Initialize CSV (header)
with open(filename, "w", newline="") as csv_datafile:
    csv_writer = csv.writer(csv_datafile)
    csv_writer.writerow(columns)

print(f"Listening for UDP data on {ip}:{port}...")
print("Press 's' to start saving, and 'q' to quit.")

# Key state tracking
pressed_keys = set()
is_saving = False
stop_program = False

def on_press(key):
    global is_saving, stop_program
    try:
        if key.char == 's':
            if not is_saving:
                print("Started saving data.")
                is_saving = True
        elif key.char == 'q':
            print("Exiting receiver.")
            stop_program = True
            return False  # Stop the key listener
    except AttributeError:
        pass

def on_release(key):
    pass

# Start key listener in background
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# Main loop
while not stop_program:
    data, addr = sock.recvfrom(4096)
    decoded_data = data.decode("utf-8")
    print(f"Received from {addr}: {decoded_data}")

    if is_saving:
        timestamp = datetime.now().isoformat('_')
        with open(filename, "a", newline="") as csv_datafile:
            csv_writer = csv.writer(csv_datafile)
            csv_writer.writerow([timestamp, decoded_data])

# Cleanup
listener.stop()
sock.close()
