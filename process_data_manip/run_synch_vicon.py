import serial
from pynput import keyboard

# Replace with your Arduino's serial port
ser = serial.Serial('/dev/ttyUSB0', 9600)

print("Press 's' to set pin HIGH, and 'q' to set pin LOW.")
print("Press ESC to exit.")

def on_press(key):
    try:
        if key.char == 's':
            ser.write(b'1')
            print("Pin set to HIGH")
        elif key.char == 'q':
            ser.write(b'0')
            print("Pin set to LOW")
    except AttributeError:
        # Handle special keys like ESC
        if key == keyboard.Key.esc:
            print("Exiting...")
            return False  # Stop listener

def on_release(key):
    pass  # You can handle key release here if needed

try:
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
except KeyboardInterrupt:
    print("Program terminated by user.")
finally:
    ser.close()
