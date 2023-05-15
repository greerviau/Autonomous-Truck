import numpy as np
import pandas as pd
from xbox_controller import XboxController
from utils import grab_frame
import pygame
import cv2
import time
import csv
import os
import sys

def main(collecting = True):
    
    resolution = (1280, 720)
    fps = 30
    pygame.init()

    joystick_count = pygame.joystick.get_count()
    print('Number of joysticks: {}'.format(joystick_count))

    js = pygame.joystick.Joystick(1)
    js.init()
    axis = js.get_axis(1)
    jsInit = js.get_init()
    jsId = js.get_id()
    jsName = js.get_name()
    print('Joystick ID: {} | Name: {} | Init status: {} | Axis(1): {}'.format(jsId, jsName, jsInit, axis))

    steer = 0
    try:
        while 1:

            frame = grab_frame(region = (0,0,1920,1080))
            frame = cv2.resize(frame, resolution)

            for event in pygame.event.get(pygame.JOYAXISMOTION):
                if event.axis == 0:
                    steer = float('{:.2f}'.format(event.value))
            
            cv2.putText(frame, 'Steer: {}'.format(steer), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow('window',cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        pygame.quit()
        print('Exiting')

if __name__ == '__main__':
    main(collecting = True)