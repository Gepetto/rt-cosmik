import socket
import os
import csv
from datetime import datetime
from multiprocessing import Process, Value, Event, Queue,Array
import logging
import time
# Set up logging for the module
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

class UDPDataSaverProcess(Process):
    def __init__(self, ip: str, port: int, output_dir: str, stop_event: Event, saving_flag: Value):
        """
        :param ip: IP address to listen on.
        :param port: UDP port number.
        :param output_dir: Directory where the CSV file will be saved.
        :param stop_event: A multiprocessing Event that signals shutdown.
        :param saving_flag: A shared boolean Value. When True, saving is active.
        """
        super().__init__()
        self.ip = ip
        self.port = port
        self.output_dir = output_dir
        self.stop_event = stop_event
        self.saving_flag = saving_flag

        # Ensure output directory exists and write CSV header once.
        os.makedirs(self.output_dir, exist_ok=True)
        self.filename = os.path.join(self.output_dir, "mks_pose.csv")
        with open(self.filename, "w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["time", "marker_data"])

    def run(self):
        logger = logging.getLogger(f"UDPDataSaverProcess-{self.pid}")
        logger.info(f"Starting UDPDataSaverProcess on {self.ip}:{self.port}")
        
        # Create and bind the UDP socket.
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.ip, self.port))
        logger.info("UDPDataSaverProcess is now listening for data...")

        # Set a timeout so that we can periodically check the stop event.
        sock.settimeout(1.0)
        
        while not self.stop_event.is_set():
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except Exception as e:
                logger.error(f"Error receiving UDP data: {e}")
                break

            decoded_data = data.decode("utf-8")
            logger.info(f"Received from {addr}: {decoded_data}")

            # Check the shared saving flag; if active, save the data.
            if self.saving_flag.value:
                timestamp = datetime.now().isoformat('_')
                with open(self.filename, "a", newline="") as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow([timestamp, decoded_data])

        sock.close()
        logger.info("UDPDataSaverProcess terminated.")



class UDPDataSaver(Process):
    def __init__(self, 
                 saving_flag: Value,  # Saving flag to control saving
                 udp_data_buffer: Array,  # Shared buffer for UDP data
                 save_dir: str,  # Directory to save CSV
                 stop_event: Event):  # Event to stop the process
        super().__init__()
        self.saving_flag = saving_flag
        self.udp_data_buffer = udp_data_buffer
        self.save_dir = save_dir
        self.stop_event = stop_event
        self.csv_file = None
        self.csv_writer = None
        self.last_save_time = time.monotonic()

    def run(self):
        try:
            # Open CSV file to save data
            self.csv_file = open(f"{self.save_dir}/mks_data.csv", mode='w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(['Timestamp', 'UDP Data'])  # CSV header
            
            while not self.stop_event.is_set():
                if self.saving_flag.value:
                    # Read the UDP data from the buffer
                    udp_data = ''.join(chr(self.udp_data_buffer[i]) for i in range(len(self.udp_data_buffer)))
                    
                    if udp_data:  # If there is any UDP data to save
                        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        # Write the UDP data and timestamp to the CSV file
                        self.csv_writer.writerow([timestamp_str, udp_data])
                        print(f"Saving UDP data: {udp_data}")

                # Sleep briefly to prevent excessive CPU usage
                time.sleep(0.1)

        except Exception as e:
            print(f"Error in DataSaverProcess: {e}")
        
        finally:
            if self.csv_file:
                self.csv_file.close()
            print("DataSaverProcess terminated.")