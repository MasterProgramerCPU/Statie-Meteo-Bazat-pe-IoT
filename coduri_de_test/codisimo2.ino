#include <Wire.h>

// APDS-9930 I2C address
#define APDS9930_I2C_ADDR 0x39

// APDS-9930 register addresses
#define APDS9930_ENABLE 0x80
#define APDS9930_ATIME 0x81
#define APDS9930_PTIME 0x82
#define APDS9930_WTIME 0x83
#define APDS9930_CONTROL 0x8F
#define APDS9930_PDATA 0x9C

// APDS-9930 command register
#define APDS9930_CMD 0xA0

// Pin for the light intensity sensor
#define LIGHT_SENSOR_PIN 34

// Function to initialize the APDS-9930 sensor
void initAPDS9930() {
  // Power on the sensor
  Wire.beginTransmission(APDS9930_I2C_ADDR);
  Wire.write(APDS9930_ENABLE);
  Wire.write(0x03);
  Wire.endTransmission();

  // Set default integration time for proximity
  Wire.beginTransmission(APDS9930_I2C_ADDR);
  Wire.write(APDS9930_PTIME);
  Wire.write(0xFF);
  Wire.endTransmission();

  // Set default wait time
  Wire.beginTransmission(APDS9930_I2C_ADDR);
  Wire.write(APDS9930_WTIME);
  Wire.write(0xFF);
  Wire.endTransmission();

  // Set control register (0x20 = Proximity gain 1x, IR LED current 100mA)
  Wire.beginTransmission(APDS9930_I2C_ADDR);
  Wire.write(APDS9930_CONTROL);
  Wire.write(0x20);
  Wire.endTransmission();
}

// Function to read proximity data from the APDS-9930 sensor
uint16_t readProximity() {
  Wire.beginTransmission(APDS9930_I2C_ADDR);
  Wire.write(APDS9930_CMD | APDS9930_PDATA);
  Wire.endTransmission();
  
  Wire.requestFrom(APDS9930_I2C_ADDR, 2);
  uint16_t proximity = Wire.read() | (Wire.read() << 8);
  
  return proximity;
}

// Function to read light intensity from the analog sensor
int readLightIntensity() {
  return analogRead(LIGHT_SENSOR_PIN);
}

void setup() {
  Serial.begin(9600);
  Wire.begin();
  
  initAPDS9930();
  delay(100);  // Allow some time for the sensor to initialize
}

void loop() {
  // Read proximity value
  uint16_t proximity = readProximity();

  // Read light intensity value
  int lightIntensity = readLightIntensity();

  // Check conditions and print result
  if (proximity < 100 && lightIntensity < 200) {
    Serial.println("Ceata : 1");
  } else {
    Serial.println("Ceata : 0");
  }

  delay(10000);  // Delay between readings
}
