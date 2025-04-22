import socket
import os
import csv
from datetime import datetime
from multiprocessing import Process, Array, Value, Lock, Barrier, Event, Queue
import logging
import time
import struct

# Set up logging for the module
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

class UDPReceiver(Process):
    def __init__(self, shared_buffer: Array,
                 timestamp_buffer: Array, # Character array for timestamp
                 lock: Lock,
                 ip: str, port: int, output_dir: str, stop_event: Event, saving_flag: Value, markers_names):
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
        self.markers_names =markers_names

        self.shared_buffer = shared_buffer
        self.timestamp_buffer = timestamp_buffer  # For timestamp string
        self.lock = lock

        # Validate timestamp buffer size (need 26 chars for format)
        if len(timestamp_buffer) != 26:
            raise ValueError("Timestamp buffer must be exactly 26 characters")
       

        self.data_length = 26+4*(len(self.markers_names) * 3)

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
                received_data, addr = sock.recvfrom(self.data_length)
            except socket.timeout:
                continue
            except Exception as e:
                logger.error(f"Error receiving UDP data: {e}")
                break

            timestamp = str(received_data[:26])[2:][:-1]
            aa = bytearray(received_data[26:])
            # logger.info(f"Received from {addr}: {decoded_data}")

            unpacked = struct.unpack("<" + "f" * (len(self.markers_names) * 3), aa)
            with self.lock:
                self.timestamp_buffer[:26] = timestamp.ljust(26, '\0').encode('utf-8')

                # copy floats
                for i, v in enumerate(unpacked):
                    self.shared_buffer[i] = v
                    # print(v)
                    

        sock.close()
        logger.info("UDPDataSaverProcess terminated.")



class UDPDataSaver(Process):
    def __init__(self, 
                 saving_flag: Value,  # Saving flag to control saving
                 shared_ts_udp,
                 shared_values_udp,
                 lock_udp,
                 cam_event, 
                 save_dir: str,  # Directory to save CSV
                 stop_event: Event):  # Event to stop the process
        super().__init__()
        self.saving_flag = saving_flag
        self.save_dir = save_dir
        self.stop_event = stop_event
        self.csv_file = None
        self.csv_writer = None
        self.last_save_time = time.monotonic()

        self.shared_ts_udp=shared_ts_udp
        self.shared_values_udp=shared_values_udp
        self.lock_udp=lock_udp
        self.cam_event=cam_event 

    def run(self):
        try:
            # Open CSV file to save data
            self.csv_file = open(f"{self.save_dir}/mks_data.csv", mode='w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(['Timestamp', 'UDP Data'])  # CSV header
            
            while not self.stop_event.is_set():
                self.cam_event.wait()
                
                with self.lock_udp:
                    ts_udp = bytes(self.shared_ts_udp[:]).decode().strip('\x00')
                    vals = list(self.shared_values_udp[:]) 
                
                self.cam_event.clear()

                if self.saving_flag.value:
                    self.csv_writer.writerow([ts_udp]+vals)


        except Exception as e:
            print(f"Error in DataSaverProcess: {e}")
        
        finally:
            if self.csv_file:
                self.csv_file.close()
            print("DataSaverProcess terminated.")