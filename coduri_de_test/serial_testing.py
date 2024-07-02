import serial
from datetime import datetime

# Define the serial ports and baud rate for both ESPs
ESP1_SERIAL_PORT = '/dev/ttyUSB0'  # Update with your actual serial port name for the first ESP
ESP2_SERIAL_PORT = '/dev/ttyUSB1'  # Update with your actual serial port name for the second ESP
BAUD_RATE = 9600

# Variables to store received data
esp1_data = None
esp2_data = None

def open_serial_connection(port):
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
        print(f"Connected to {port} at {BAUD_RATE} baud.")
        return ser
    except serial.SerialException as e:
        print(f"Failed to connect to {port}: {e}")
        return None

def read_serial(ser1, ser2):
    global esp1_data, esp2_data

    while True:
        if ser1 and ser1.in_waiting > 0:
            line = ser1.readline().decode('utf-8').rstrip()
            print(f"Received from ESP1: {line}")
            esp1_data = line
            if esp1_data and esp2_data:
                process_data()
                esp1_data, esp2_data = None, None
        
        if ser2 and ser2.in_waiting > 0:
            line = ser2.readline().decode('utf-8').rstrip()
            print(f"Received from ESP2: {line}")
            esp2_data = line
            if esp1_data and esp2_data:
                process_data()
                esp1_data, esp2_data = None, None

def process_data():
    global esp1_data, esp2_data

    if esp1_data:
        process_esp1_data(esp1_data)
    if esp2_data:
        process_esp2_data(esp2_data)

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
    ser1 = open_serial_connection(ESP1_SERIAL_PORT)
    ser2 = open_serial_connection(ESP2_SERIAL_PORT)

    if ser1 or ser2:
        read_serial(ser1, ser2)
    else:
        print("Failed to connect to any serial ports.")
