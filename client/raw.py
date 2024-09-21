import time

from api import RobotAPI as rapi

robot = rapi.RobotAPI(flag_serial=False)
robot.set_camera(100, 640, 320)
fps = 0
fps_count = 0
t = time.time()
while 1:
    fps_count += 1
    if time.time() > t + 1:
        fps = fps_count
        fps_count = 0
        t = time.time()

    frame = robot.get_frame(wait_new_frame=False)
    # frame = cv2.flip(frame, 1)
    robot.text_to_frame(frame, "FPS: " + str(fps), 20, 20)
    robot.set_frame(frame, 40)
