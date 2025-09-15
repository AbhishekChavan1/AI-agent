// Include the necessary libraries
#include <Arduino.h>

// Define the pins for the ultrasonic sensor and LED
const int ultrasonic_trig = 2;
const int ultrasonic_echo = 3;
const int led_pin = 13;

// Define the maximum distance for the ultrasonic sensor
const int max_distance = 200;

void setup() {
  // Initialize the serial communication at 9600 bps
  Serial.begin(9600);
  // Initialize the LED pin as an output
  pinMode(led_pin, OUTPUT);
  // Initialize the ultrasonic sensor pins
  pinMode(ultrasonic_trig, OUTPUT);
  pinMode(ultrasonic_echo, INPUT);
}

void loop() {
  // Get the distance from the ultrasonic sensor
  long duration, distance;
  digitalWrite(ultrasonic_trig, LOW);
  delayMicroseconds(2);
  digitalWrite(ultrasonic_trig, HIGH);
  delayMicroseconds(10);
  digitalWrite(ultrasonic_trig, LOW);
  duration = pulseIn(ultrasonic_echo, HIGH);
  distance = (duration / 2) / 29.1;

  // Print the distance to the serial monitor
  Serial.print("Distance: ");
  Serial.print(distance);
  Serial.println(" cm");

  // Turn the LED on if the distance is less than 10 cm
  if (distance < 10) {
    digitalWrite(led_pin, HIGH);
  } else {
    digitalWrite(led_pin, LOW);
  }

  // Wait for 50 milliseconds before taking the next reading
  delay(50);
}