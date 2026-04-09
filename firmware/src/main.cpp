#include <Arduino.h>  // Mandatory for PlatformIO
#include <ESP32Servo.h>

/* * Kart Receiver Test
 * Purpose: Receive "S90T120" packets from Python and print the results.
 */

void setup() {
  // Must match the baudrate in your hardware.py
  Serial.begin(115200);
  while (!Serial) { ; } // Wait for connection
  
  Serial.println("--- ESP32 KART RECEIVER ONLINE ---");
  Serial.println("Awaiting packets from Python...");
}

void loop() {
  // Check if data is waiting in the buffer
  if (Serial.available() > 0) {
    // Read the string until the newline character '\n'
    String data = Serial.readStringUntil('\n');
    
    // Basic validation: Check if it starts with 'S' and contains 'T'
    if (data.startsWith("S") && data.indexOf('T') != -1) {
      int tIndex = data.indexOf('T');
      
      // Extract Steering (from char 1 to T)
      String steerString = data.substring(1, tIndex);
      // Extract Throttle (from T+1 to end)
      String throttleString = data.substring(tIndex + 1);

      // Convert to integers
      int steerVal = steerString.toInt();
      int throttleVal = throttleString.toInt();

      // Print back for verification
      Serial.print("RECEIVING -> Steering: ");
      Serial.print(steerVal);
      Serial.print(" | Throttle: ");
      Serial.println(throttleVal);
    } 
    else {
      // Ignore junk data or partial packets
      Serial.print("MALFORMED PACKET: ");
      Serial.println(data);
    }
  }
}
// pio run -t upload