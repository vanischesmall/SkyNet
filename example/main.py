from api import RobotAPI
import serial, time

uart = serial.Serial("/dev/ttyS3", baudrate=9600, stopbits=serial.STOPBITS_ONE)

robot = RobotAPI.RobotAPI(flag_serial=False)
robot.set_camera(120, 640, 480)

frames, frames_cnt, frames_tmr = 0, 0, 0
tim = time.time()


def fps():
    """
    Updates the frames per second count.
    """
    global frames, frames_cnt, frames_tmr

    if tim > frames_tmr + 1:
        frames = frames_cnt
        frames_cnt, frames_tmr = 0, tim


def write_uart(package):
    """
    Writes the given package to the UART connection.

    Args:
        package (str): The package to be written.
    """
    uart.write(package.encode("utf-8"))
    uart.reset_output_buffer()


def read_uart():
    """
    Reads data from the UART connection.

    Returns:
        str: The received package.
    """
    if uart.in_waiting > 0:
        pkg, dead_tmr = '', tim
        while True:
            char = str(uart.read(), 'utf-8')

            if char == ";":
                return pkg
            if tim > dead_tmr + 0.02:
                break

            pkg += char
        uart.reset_input_buffer()


if __name__ == "__main__":
    frame = robot.get_frame(wait_new_frame=1)  # Get a frame from the robot's camera

    tim = time.time()  # Update the current time
    message = 'Hello, average SkyNet enjoyer!'  # Define a message to send via UART

    write_uart(message)  # Send the message via UART

    fps()  # Update the frames per second count
    robot.text_to_frame(frame, frames, 150, 20)  # Add text to the frame, displaying the frames per second count
    robot.set_frame(frame, 40)  # Set the frame on the robot's display with a brightness level of 40