import cv2
import numpy as np

def get_curve(rng, x_d, y_d):
    point_list = []
    j = 1/100
    for i in range(100):
        t = float(j*i)*(np.pi/2)
        #print(t)
        x = np.cos(t)*(rng*100)
        
        y = np.sin(t)*150
        X = x+x_d
        if rng < 0:
            X+=abs(rng*100)
        else:
            X-=abs(rng*100)
        Y = y_d-y
        point_list.append((X, Y))
    print(point_list)
    pts = np.array(point_list).astype(np.int32)
    return pts

origin = 0
end = 1

pts_left = get_curve(0.1, 250,500)
pts_right = get_curve(-0.6, 250,500)

frame = np.zeros((500,500,3))
cv2.polylines(frame,[pts_left],False,(0,255,0),10)
cv2.polylines(frame,[pts_right],False,(0,255,0),10)
cv2.imshow('curve', frame)
cv2.waitKey(0)
