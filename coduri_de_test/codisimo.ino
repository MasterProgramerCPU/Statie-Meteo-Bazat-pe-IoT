// Include necessary libraries
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BMP085_U.h>
#include <DHT.h>

// Define sensor pins
#define DHTPIN 4
#define DHTTYPE DHT22
#define MOTORPIN 13

// Create sensor objects
DHT dht(DHTPIN, DHTTYPE);
Adafruit_BMP085_Unified bmp = Adafruit_BMP085_Unified(180);

// Function to initialize BMP180 sensor
void initializeBMP180() {
  if (!bmp.begin()) {
    Serial.println("Could not find a valid BMP180 sensor, check wiring!");
    while (1) {}
  }
}

void setup() {
  // Initialize serial communication
  Serial.begin(9600);

  // Initialize DHT22 sensor
  dht.begin();

  // Initialize BMP180 sensor
  initializeBMP180();

  // Configure motor pin as input
  pinMode(MOTORPIN, INPUT);
}

void loop() {
  // Read data from DHT22
  float humidity = dht.readHumidity();
  float temperature = dht.readTemperature();

  // Check if any reads failed and exit early (to try again).
  if (isnan(humidity) || isnan(temperature)) {
    Serial.println("Failed to read from DHT sensor!");
    return;
  }

  // Read data from BMP180
  sensors_event_t event;
  bmp.getEvent(&event);
  float pressure = 0;
  if (event.pressure) {
    pressure = event.pressure;
  }

  float altitude = bmp.pressureToAltitude(101325, pressure); // assuming sea level pressure of 101325 Pa

  // Read data from motor (analog value)
  int motorValue = analogRead(MOTORPIN);

  // Send data through serial in a parsable format
  Serial.print("Temperatura: ");
  Serial.print(temperature);
  Serial.print(" C, Umiditate: ");
  Serial.print(humidity);
  Serial.print(" %, Presiune: ");
  Serial.print(pressure);
  Serial.print(" Pa, Anemometru: ");
  Serial.print(motorValue);
  Serial.print("\n");
  // Wait for an hour (3600000 milliseconds)
  delay(60000);
}
