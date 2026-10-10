/*
  APNVI ESP32: the forward ultrasonic (Stage 1) plus the hold-to-talk button (Stage 2).

  Replaces the Stage 1 sketch esp32/ultrasonic/ultrasonic.ino. The ultrasonic part keeps
  the agreed Stage 1 settings exactly (stage1-weekend-plan.md):
    115200 baud, one number per line, metres with 2 decimals (e.g. 1.23),
    -1 when nothing is in range, about 10 times a second, TRIG on GPIO 5, ECHO on GPIO 18.

  New for Stage 2 (decided 2026-10-08: hold to talk):
    one push button between GPIO 23 and GND. No resistor: the ESP32's built-in one is used.
    Pressed  -> sends a line "B1"
    Released -> sends a line "B0"
  The Stage 1 reader skips any line that is not a number, so it ignores these lines.
*/

const int TRIG_PIN = 5;
const int ECHO_PIN = 18;
const int BUTTON_PIN = 23;

const unsigned long BAUD = 115200;
const unsigned long DISTANCE_EVERY_MS = 100;    // about 10 distances a second
const unsigned long ECHO_TIMEOUT_US = 25000;    // no echo after 25 ms = nothing within ~4 m
const float SOUND_M_PER_US = 0.000343;          // speed of sound, metres per microsecond
const unsigned long DEBOUNCE_MS = 30;           // ignore the contact bouncing when pressed

unsigned long lastDistanceMs = 0;
bool buttonDown = false;                        // what we last reported
bool lastReading = false;                       // what the pin said last time we looked
unsigned long readingSinceMs = 0;               // since when the pin has said that

float readDistanceM() {
  // Send a 10 microsecond pulse, then time how long the echo takes to come back.
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  unsigned long echoUs = pulseIn(ECHO_PIN, HIGH, ECHO_TIMEOUT_US);
  if (echoUs == 0) {
    return -1.0;                                // no echo: nothing in range
  }
  return echoUs * SOUND_M_PER_US / 2.0;         // there and back, so halve it
}

void checkButton() {
  // The pin reads LOW while the button is pressed (it connects the pin to GND).
  bool reading = digitalRead(BUTTON_PIN) == LOW;
  unsigned long now = millis();
  if (reading != lastReading) {
    lastReading = reading;
    readingSinceMs = now;
  }
  // Only believe a change once it has held still for DEBOUNCE_MS.
  if (reading != buttonDown && now - readingSinceMs >= DEBOUNCE_MS) {
    buttonDown = reading;
    Serial.println(buttonDown ? "B1" : "B0");
  }
}

void setup() {
  Serial.begin(BAUD);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  digitalWrite(TRIG_PIN, LOW);
}

void loop() {
  checkButton();
  unsigned long now = millis();
  if (now - lastDistanceMs >= DISTANCE_EVERY_MS) {
    lastDistanceMs = now;
    float distance = readDistanceM();
    if (distance < 0) {
      Serial.println("-1");
    } else {
      Serial.println(distance, 2);
    }
    checkButton();                              // the echo wait can take up to 25 ms
  }
}
