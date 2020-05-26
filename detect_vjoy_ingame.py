import pyvjoy

joy = pyvjoy.VJoyDevice(1)

while True:
    joy.update()