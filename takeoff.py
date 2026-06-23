import tello
import time

tello.start()

print("Taking off...")
tello.takeoff()
print("Drone is airborne.")
time.sleep(2)

# Move forward
print("Moving forward...")
tello.forward(100)
time.sleep(1)

# Fly a square circuit (forward -> right -> right -> right -> right = full circle back)
print("Turning and circling back...")
for i in range(4):
    tello.clockwise(90)
    time.sleep(0.5)
    tello.forward(50)
    time.sleep(1)

print("Landing...")
tello.land()
print("Done.")
