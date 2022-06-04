

import serial
ser = serial.Serial('COM5', 9800, timeout=1)

while True:
    ser.write(b'H')
