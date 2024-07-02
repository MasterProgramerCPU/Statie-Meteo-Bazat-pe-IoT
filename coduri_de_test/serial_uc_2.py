import serial
from datetime import datetime

# Define the serial port and baud rate for the second ESP
ESP2_SERIAL_PORT = '/dev/ttyUSB1'  # Update with your actual serial port name for the second ESP
BAUD_RATE = 9600

def open_serial_connection(port):
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
        print(f"Connected to {port} at {BAUD_RATE} baud.")
        return ser
    except serial.SerialException as e:
        print(f"Failed to connect to {port}: {e}")
        return None

def read_serial(ser):
    while True:
        if ser and ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').rstrip()
            print(f"Received from ESP2: {line}")
            process_esp2_data(line)

def process_esp2_data(line):
    if "Ceata :" in line:
        try:
            ceata = int(line.split(": ")[1].strip())

            # Get current date and time
            now = datetime.now()
            current_date = now.strftime('%Y-%m-%d')
            current_time = now.strftime('%H%M')

            # Print SQL query for ceata
            print(f"DEBUG: Generating SQL query for ESP2 data.")
            print(f"INSERT INTO measurements_ceata (valoarea) VALUES ({ceata});")
            print(f"INSERT INTO measurements_masuratoare (dataMasuratoare, timpMasuratoare) VALUES ('{current_date}', '{current_time}');")
        except (IndexError, ValueError) as e:
            print(f"Error parsing data from ESP2: {e}")

if __name__ == '__main__':
    ser = open_serial_connection(ESP2_SERIAL_PORT)
    if ser:
        read_serial(ser)
    else:
        print("Failed to connect to the serial port.")
