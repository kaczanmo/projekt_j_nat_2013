'''


Created on 05-01-2014

@author: Tenac
'''


import serial
import time

# port=serial.Serial('COM1', 9600)
# time.sleep(1)
# # port.open
# time.sleep(1)
# # port.readline()
# port.write('data\r')
# time.sleep(1)

import serial
import time
import array
PORT = 'COM1'
BOUDR = 9600



def ConnectToSerialPort(port=PORT):
    global ser
    ser = serial.Serial(PORT, BOUDR)
    time.sleep(2)
    
def DisconnectSerialPort():
    time.sleep(1)
    ser.close()   

    
def SendCommandToSerialPort(commandText):
    for i in (range(len(commandText))): 
        ser.write(commandText[i].encode())

    ser.write(b'\r')


    
if __name__ == '__main__':    
    SendCommandToSerialPort('ok')
    
#     print (ser.readline())
#     print (ser.readline())
#     print (ser.readline())


# ser = serial.Serial('COM1', 9600, timeout=5.0)
#   
# while 1:
#  try:
#   print (ser.readline())
#   var = input("Enter 0 or 1 to control led: ")
#   ser.write(var)
#   time.sleep(1)
#  except ser.SerialTimeoutException:
#   print('Data could not be read')
#   time.sleep(1)
  
  
  
# input=1
# while 1 :
#     # get keyboard input
#     input = input('>>')
#     if input == 'exit':
#         ser.close()
#         exit()
#     else:
#         # send the character to the device
#         # (note that I happend a \r\n carriage return and line feed to the characters - this is requested by my device)
#         ser.write(input + '\r\n')
#         out = ''
#         # let's wait one second before reading output (let's give device time to answer)
#         time.sleep(1)
#         while ser.inWaiting() > 0:
#             out += ser.read(1)
#             
#         if out != '':
#             print (">>" + out)
#             
         