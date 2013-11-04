import time
import numpy as np
import matplotlib.pyplot as plt
import threading 


def go():
    i=0
    while i <10:
        temp_y=np.random.random()
        x.append(i)
        y.append(temp_y)
        plt.scatter(i,temp_y)
        i+=1
        plt.draw()
#         time.sleep(0.2)
        
if __name__ == '__main__':
    fig=plt.figure()
    plt.axis([0,1000,0,1])

    x=list()
    y=list()

    plt.ion()
    plt.show()
    threading.Thread(go()).start()
    