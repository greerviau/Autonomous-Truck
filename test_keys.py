from direct_keys import PressKey, ReleaseKey, W, A, S, D, L
import time

while True:
    PressKey(L)
    ReleaseKey(L)
    time.sleep(0.2)
    PressKey(L)
    ReleaseKey(L)
    time.sleep(0.2)
    PressKey(L)
    ReleaseKey(L)
    time.sleep(0.2)
    print('loop')