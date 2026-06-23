import tello

tello.start()

battery = tello.get_battery()
print(f"Battery level: {battery}%")

if battery < 20:
    print("Warning: Battery is low!")
elif battery < 50:
    print("Battery is moderate.")
else:
    print("Battery is sufficient.")
