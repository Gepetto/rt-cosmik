import socket
import os
import keyboard  # For key press detection
import csv
from datetime import datetime
# UDP Configuration
ip = "172.20.164.200"  # The IP the receiver listens on
port = 44445  # The port to receive data on

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((ip, port))

# CSV Setup
output_dir = "/root/workspace/ros_ws/src/rt-cosmik/output"
filename = os.path.join(output_dir, "mks_pose.csv")
columns = ["time", "marker_data"]  # You can expand this to individual marker positions if needed

# Initialize CSV (header)
with open(filename, "w", newline="") as csv_datafile:
    csv_writer = csv.writer(csv_datafile)
    csv_writer.writerow(columns)

print(f"Listening for UDP data on {ip}:{port}...")

# Variable to track if we are saving data
is_saving = False

while True:
    if keyboard.is_pressed('s'):  # Check if the "s" key is pressed
        if not is_saving:
            print("Started saving data.")
            is_saving = True
    elif keyboard.is_pressed('q'):  # Press 'q' to stop the program
        print("Exiting receiver.")
        break

    # if is_saving:
    data, addr = sock.recvfrom(4096)  # Buffer size 1024 bytes
    decoded_data = data.decode("utf-8")  # Decode the received bytes into a string

    print(f"Received from {addr}: {decoded_data}")  # Print received data

    # Get current timestamp
    timestamp = datetime.now().isoformat('_')

    # Save to CSV
    with open(filename, "a", newline="") as csv_datafile:
        csv_writer = csv.writer(csv_datafile)
        csv_writer.writerow([timestamp, decoded_data])  # Write timestamp and data

