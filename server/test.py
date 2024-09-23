import time

import cv2
import numpy as np
import serial

from api import RobotAPI as rapi

low_black = np.array([0, 0, 0])
up_black = np.array([0, 0, 0])

low_red = np.array([0, 0, 0])
up_red = np.array([0, 0, 0])

low_green = np.array([60, 150, 20])
up_green = np.array([90, 255, 220])

low_blue = np.array([0, 0, 0])
up_blue = np.array([0, 0, 0])

low_orange = np.array([0, 0, 0])
up_orange = np.array([0, 0, 0])


port = serial.Serial("/dev/ttyS0", baudrate=115200, stopbits=serial.STOPBITS_ONE)
robot = rapi.RobotAPI(flag_serial=False)
robot.set_camera(100, 640, 480)

message = ""
fps = 0
fps1 = 0
fps_time = 0

xb11, yb11 = 20, 290
xb21, yb21 = 190, 370

xb12, yb12 = 470, 290
xb22, yb22 = 620, 370

xg1, yg1 = 300, 350
xg2, yg2 = 340, 410

lowb = np.array([0, 0, 0])
upb = np.array([180, 180, 100])

lowg = np.array([61, 188, 46])
upg = np.array([82, 255, 256])

max1 = 0
max2 = 0
e_old = 0


def black_line():
    x1 = 0

    dat = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(dat, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, low_red, up_red)
    imd, contours, hod = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    xd, yd, wd, hd = 0, 0, 0, 0
    for contor in contours:
        x, y, w, h = cv2.boundingRect(contor)
        if h * w > 100 and h * w > hd * wd:
            xd, yd, wd, hd = x, y, w, h

    cv2.rectangle(dat, (xd, yd), (xd + wd, yd + hd), (0, 255, 255), 2)

    cv2.rectangle(frame, (xb11, yb11), (xb21, yb21), (0, 0, 255), 2)

    datb2 = frame[yb12:yb22, xb12:xb22]
    cv2.rectangle(frame, (xb12, yb12), (xb22, yb22), (0, 0, 255), 2)

    dat2 = cv2.GaussianBlur(datb2, (5, 5), cv2.BORDER_DEFAULT)
    hsv2 = cv2.cvtColor(dat2, cv2.COLOR_BGR2HSV)
    maskd2 = cv2.inRange(hsv2, lowb, upb)

    gray2 = cv2.cvtColor(maskd2, cv2.COLOR_GRAY2BGR)
    frame[yb12:yb22, xb12:xb22] = gray2

    imd2, contoursd2, hod2 = cv2.findContours(
        maskd2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    max2 = 0

    for contorb2 in contoursd2:
        x, y, w, h = cv2.boundingRect(contorb2)
        a2 = cv2.contourArea(contorb2)
        if a2 > 200:
            if a2 > max2:
                max2 = a2
            cv2.rectangle(datb2, (x, y), (x + w, y + h), (0, 255, 0), 2)


def green():
    global xg1, yg1, xg2, yg2, lowg, upg

    datg = frame[yg1:yg2, xg1:xg2]
    cv2.rectangle(frame, (xg1, yg1), (xg2, yg2), (0, 255, 0), 2)

    dat1 = cv2.GaussianBlur(datg, (5, 5), cv2.BORDER_DEFAULT)
    hsv1 = cv2.cvtColor(dat1, cv2.COLOR_BGR2HSV)
    maskd1 = cv2.inRange(hsv1, lowg, upg)

    gray = cv2.cvtColor(maskd1, cv2.COLOR_GRAY2BGR)
    frame[yg1:yg2, xg1:xg2] = gray

    imd1, contoursd1, hod1 = cv2.findContours(
        maskd1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    for contorb1 in contoursd1:
        x, y, w, h = cv2.boundingRect(contorb1)
        a1 = cv2.contourArea(contorb1)
        if a1 > 100:
            cv2.rectangle(datg, (x, y), (x + w, y + h), (0, 0, 255), 2)


kp = 2
kd = 5
z = 0
while 1:
    frame = robot.get_frame(wait_new_frame=1)
    black_line()
    e = max1 // 10 - max2 // 10
    u = e * kp + (e - e_old) * kd
    if max1 < max2:
        z = 1
    else:
        z = 2
    if max1 < 100 or max2 < 100:
        if z == 1:
            deg = 2150
        else:
            deg = 150
    else:
        deg = 1150 - u

    speed = 25
    if deg > 2150:
        deg = 2150
    if deg < 150:
        deg = 150
    # green()
    # deg speed
    # 2150     1150     150
    message = str(int(deg) + 1000) + str(int(speed) + 100) + "$"
    port.write(message.encode("utf-8"))
    robot.text_to_frame(frame, message, 20, 20)
    robot.text_to_frame(frame, str(e) + " " + str(u) + " " + str(deg), 200, 20)

    fps1 += 1
    if time.time() > fps_time + 1:
        fps_time = time.time()
        fps = fps1
        fps1 = 0

    robot.text_to_frame(frame, "fps = " + str(fps), 500, 20)
    robot.set_frame(frame, 40)

