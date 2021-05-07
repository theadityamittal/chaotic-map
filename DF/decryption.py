import cv2
import numpy as np
from PIL import Image
from numpy import *

def decodeKey(rows, cols):
    f = open("key.txt", "r")
    key = [[0 for i in range(rows)] for j in range(cols)]
    for i in range(rows):
        for j in range(cols):
            key[i][j] = int(f.readline())
    return key

def deshuffleMagicSquare(n, img):
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
    
    temp = cv2.imread("decrypted.png")
    for i in range(n):
        for j in range(n):
            rowindex = (arr[i][j] - 1)%n
            colindex = int((arr[i][j] - 1)/n)
            temp[rowindex][colindex] = img[i][j]
    return temp

def decrypt(img, key):
    new_image = array(img)
    N = new_image.shape[0]
    x,y = meshgrid(range(N),range(N))
    xmap = (x + y) % N
    ymap = (x + 2*y) % N
    for i in range(1 + N - int(N/16)):
        new_image = new_image[xmap,ymap]
    for i in range(N):
        for j in range(N):
            for k in range(3):
                new_image[i][j][2-k] = new_image[i][j][2-k] ^ key[i][j]
    im = Image.fromarray(new_image)
    arr = np.array(im, dtype=np.uint8)
    result = Image.fromarray(arr)
    result.save('decrypted.png')
    cipher = cv2.imread("decrypted.png")
    decipher = deshuffleMagicSquare(N, cipher)
    cv2.imwrite('decrypted.png', decipher)
    return decipher

if __name__ == '__main__':
    image = cv2.imread("encrypted.png")
    N = array(image).shape[0]
    key = decodeKey(N, N)
    decipher = decrypt(image, key)