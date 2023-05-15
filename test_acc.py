from tensorflow.keras.models import load_model

from xbox_controller import XboxController
from utils import get_speed
from utils import grab_frame
from direct_keys import PressKey, ReleaseKey, W, A, S, D

from PIL import ImageGrab

import cv2

ocr_model = load_model('digit_rec.h5')

xbox = XboxController()

human_override=True

lock = False

while True:
    frame = grab_frame(region = (0,0,1920,1080))
    frame = cv2.resize(frame, (1280, 720))

    input_dict = xbox.read()

    left_b = input_dict['LB']

    if left_b == 1 and not lock:
        if human_override:
            human_override=False
        elif not human_override:
            human_override=True
        
        lock = True

    if left_b == 0 and lock:
        lock = False

    speed, speed_limit = get_speed(frame, ocr_model)

    ReleaseKey(W)
    ReleaseKey(S)

    if not human_override:

        if speed < speed_limit:
            PressKey(W)

    print('Human Control {} - Speed Limit: {} - Speed: {} '.format(str(human_override), speed_limit, speed))

    