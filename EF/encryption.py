import cv2
import numpy as np
import random
from PIL import Image
from numpy import *
import time
# import pyRAPL

# pyRAPL.setup() 
# pyRAPL.setup(devices=[pyRAPL.Device.PKG], socket_ids=[1])

def shuffleMagicSquare(n, img, filename):
    arr = [[(n*y)+x+1 for x in range(n)]for y in range(n)]
    for i in range(0,int(n/4)):
    	for j in range(0,int(n/4)):
    		arr[i][j] = (n*n + 1) - arr[i][j]
	
    for i in range(0,int(n/4)):
	    for j in range(3 * (int(n/4)),n):
		    arr[i][j] = (n*n + 1) - arr[i][j]

    for i in range(3 * (int(n/4)),n):
    	for j in range(0,int(n/4)):
    		arr[i][j] = (n*n + 1) - arr[i][j]
	
    for i in range(3 * (int(n/4)),n):
    	for j in range(3 * (int(n/4)),n):
    		arr[i][j] = (n*n + 1) - arr[i][j]
			
    for i in range(int(n/4),3 * (int(n/4))):
    	for j in range(int(n/4),3 * (int(n/4))):
    		arr[i][j] = (n*n + 1) - arr[i][j]
    
    temp = cv2.imread("../test images/" + filename)
    for i in range(n):
        for j in range(n):
            rowindex = (arr[i][j] - 1)%n
            colindex = int((arr[i][j] - 1)/n)
            temp[i][j] = img[rowindex][colindex]
    return temp


def generateKey(rows, cols):
    R = 3.99
    X = 0.4 + random.random()/5
    f = open("../DF/key.txt", "w")
    key = [[0 for i in range(rows)] for j in range(cols)]
    for i in range(rows):
        for j in range(cols):
            val = 1 - X
            X = X * R * val
            key[i][j] = (int(X * 255))
            f.write(str(key[i][j]) + "\n")
    return key

# @pyRAPL.measure
def encrypt(img, key, filename):
    N = img.shape[0]
    shuffledimage = shuffleMagicSquare(N, img, filename)
    for i in range(N):
        for j in range(N):
            for  k in range(3):
                shuffledimage[i][j][k] = shuffledimage[i][j][k] ^ key[i][j]
    array = np.array(shuffledimage, dtype=np.uint8)
    x,y = meshgrid(range(N),range(N))
    xmap = (x + y) % N
    ymap = (x + 2*y) % N
    for i in range(int(N/16)):
        result = Image.fromarray(array)
        array = array[xmap,ymap]
    result.save("../DF/encrypted.png")
    result.save("cipher.png")
    

if __name__ == '__main__':
    start_time = time.time()
    #filename = input("Enter File Name (with extension): ")
    filename = 'lena/10.png'
    image = cv2.imread("../test images/" + filename)
    print(image.shape)
    rows, cols, num = image.shape
    key = generateKey(rows, cols)
    encrypt(image, key, filename)
    print("--- %s seconds ---" % (time.time() - start_time))