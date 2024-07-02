import serial
from datetime import datetime

# Define the serial port and baud rate for the first ESP
ESP1_SERIAL_PORT = '/dev/ttyUSB0'  # Update with your actual serial port name for the first ESP
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
            print(f"Received from ESP1: {line}")
            process_esp1_data(line)

def process_esp1_data(line):
    if "Temperatura:" in line and "Umiditate:" in line and "Presiune:" in line and "Anemometru:" in line:
        try:
            parts = line.split(", ")
            temperature = float(parts[0].split(": ")[1].strip(" C"))
            humidity = float(parts[1].split(": ")[1].strip(" %"))
            pressure = float(parts[2].split(": ")[1].strip(" Pa"))
            anemometer = int(parts[3].split(": ")[1].strip())

            # Get current date and time
            now = datetime.now()
            current_date = now.strftime('%Y-%m-%d')
            current_time = now.strftime('%H%M')

            # Print SQL queries
            print(f"DEBUG: Generating SQL queries for ESP1 data.")
            print(f"INSERT INTO measurements_anemometru (valoarea_medie) VALUES ({anemometer});")
            print(f"INSERT INTO measurements_temperatura (valoarea_medie) VALUES ({temperature});")
            print(f"INSERT INTO measurements_presiune (valoare) VALUES ({pressure});")
            print(f"INSERT INTO measurements_umiditate (valoarea_medie) VALUES ({humidity});")
            print(f"INSERT INTO measurements_masuratoare (dataMasuratoare, timpMasuratoare) VALUES ('{current_date}', '{current_time}');")
        except (IndexError, ValueError) as e:
            print(f"Error parsing data from ESP1: {e}")

if __name__ == '__main__':
    ser = open_serial_connection(ESP1_SERIAL_PORT)
    if ser:
        read_serial(ser)
    else:
        print("Failed to connect to the serial port.")
