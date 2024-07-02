#======================================================================================================================================================
#Autor : Geana Tudor-Bogdan
#Nume Fisier : reading_serial_port
#Descriere : Cod folosit pentru a citi de pe portul serial datele masurate si a le stoca intr-un dictionar(este citita o singura intrare)
#Data :
#=======================================================================================================================================================

import serial  # type: ignore

#functia utilizata pentru a inlatura caracterele asociate transmiterii seriale
def clean_line(line):
	space = ''
	line = ser.readline()
	line = str(line)
	space = space.join(e for e in line if e.isalnum())
	space = space.strip('brn')

	return space

#instantierea conexiunii seriale 
ser = serial.Serial('/dev/ttyUSB0',
                    baudrate=9600,
					parity=serial.PARITY_NONE,
					stopbits=serial.STOPBITS_ONE)

#declararea dictionarului care ba stoca intrarea de date
measuredict = {
	"numar_masuratoare" : '0',
	"temperatura" : '0',
	"presiune" : '0',
	"ceata" : '0',
	"anemometru" : '0',
	"umiditate" : '0',
	"lumina" : '0'
}

#parsarea intrarii seriale cu scopul de a extrage datele masurate
i = 0
for i in range(11):

	space = ''
	line = ser.readline()
	line = str(line)
	space = space.join(e for e in line if e.isalnum())
	space = space.strip('brn')
	print(space)

	if 'Masuratoarea' in line:
		l = clean_line(line) 
		print(l)
		measuredict['numar_masuratoare'] = l

	if 'Temperatura' in line:
		l = clean_line(line)
		print(l)
		measuredict['temperatura'] = l

	if 'Presiune' in line:
		l = clean_line(line)
		print(l)
		measuredict['presiune'] = l

	if 'Ceata' in line:
		l = clean_line(line)
		print(l)
		measuredict['ceata'] = l

	if 'Anemometru' in line:
		l = clean_line(line)
		print(l)
		measuredict['anemometru'] = l

	if 'Umiditate' in line:
		l = clean_line(line)
		print(l)
		measuredict['umiditate'] = l

	if 'Lumina' in line:
		l = clean_line(line)
		print(l)
		measuredict['lumina'] = l	


#afisarea dictionarului
for y in measuredict:
	print(y, ":",measuredict[y])	

		
	
		


#=======================================================================================================================================================