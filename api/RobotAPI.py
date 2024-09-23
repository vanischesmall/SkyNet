__author__ = "Yuri Glamazdin <yglamazdin@gmail.com>"
__version__ = "1.6"

import atexit
import socket
import struct
import threading

# install turbo
# sudo apt-get install libturbojpeg-dev
import time

import cv2
import zmq


# for TURBO
def cv2_decode_image_buffer(img_buffer):
    img_array = np.frombuffer(img_buffer, dtype=np.dtype("uint8"))
    # Decode a colored image
    return cv2.imdecode(img_array, flags=cv2.IMREAD_UNCHANGED)


def cv2_encode_image(cv2_img, jpeg_quality):
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    result, buf = cv2.imencode(".jpg", cv2_img, encode_params)
    return buf.tobytes()


def turbo_decode_image_buffer(img_buffer, jpeg):
    return jpeg.decode(img_buffer)


def turbo_encode_image(cv2_img, jpeg, jpeg_quality):
    return jpeg.encode(cv2_img, quality=jpeg_quality)


import math
import os
import platform
import warnings
from ctypes import *
from ctypes.util import find_library

import numpy as np

# default libTurboJPEG library path
DEFAULT_LIB_PATHS = {
    "Darwin": ["/usr/local/opt/jpeg-turbo/lib/libturbojpeg.dylib"],
    "Linux": [
        "/usr/lib/x86_64-linux-gnu/libturbojpeg.so.0",
        "/opt/libjpeg-turbo/lib64/libturbojpeg.so",
    ],
    "Windows": ["C:/libjpeg-turbo-gcc64/bin/libturbojpeg.dll"],
}

# error codes
# see details in https://github.com/libjpeg-turbo/libjpeg-turbo/blob/master/turbojpeg.h
TJERR_WARNING = 0
TJERR_FATAL = 1

# color spaces
# see details in https://github.com/libjpeg-turbo/libjpeg-turbo/blob/master/turbojpeg.h
TJCS_RGB = 0
TJCS_YCbCr = 1
TJCS_GRAY = 2
TJCS_CMYK = 3
TJCS_YCCK = 4

# pixel formats
# see details in https://github.com/libjpeg-turbo/libjpeg-turbo/blob/master/turbojpeg.h
TJPF_RGB = 0
TJPF_BGR = 1
TJPF_RGBX = 2
TJPF_BGRX = 3
TJPF_XBGR = 4
TJPF_XRGB = 5
TJPF_GRAY = 6
TJPF_RGBA = 7
TJPF_BGRA = 8
TJPF_ABGR = 9
TJPF_ARGB = 10
TJPF_CMYK = 11

# chrominance subsampling options
# see details in https://github.com/libjpeg-turbo/libjpeg-turbo/blob/master/turbojpeg.h
TJSAMP_444 = 0
TJSAMP_422 = 1
TJSAMP_420 = 2
TJSAMP_GRAY = 3
TJSAMP_440 = 4
TJSAMP_411 = 5

# miscellaneous flags
# see details in https://github.com/libjpeg-turbo/libjpeg-turbo/blob/master/turbojpeg.h
# note: TJFLAG_NOREALLOC cannot be supported due to reallocation is needed by PyTurboJPEG.
TJFLAG_BOTTOMUP = 2
TJFLAG_FASTUPSAMPLE = 256
TJFLAG_FASTDCT = 2048
TJFLAG_ACCURATEDCT = 4096
TJFLAG_STOPONWARNING = 8192
TJFLAG_PROGRESSIVE = 16384


class TurboJPEG(object):
    """A Python wrapper of libjpeg-turbo for decoding and encoding JPEG image."""

    def __init__(self, lib_path=None):
        turbo_jpeg = cdll.LoadLibrary(
            self.__find_turbojpeg() if lib_path is None else lib_path
        )
        self.__init_decompress = turbo_jpeg.tjInitDecompress
        self.__init_decompress.restype = c_void_p
        self.__init_compress = turbo_jpeg.tjInitCompress
        self.__init_compress.restype = c_void_p
        self.__destroy = turbo_jpeg.tjDestroy
        self.__destroy.argtypes = [c_void_p]
        self.__destroy.restype = c_int
        self.__decompress_header = turbo_jpeg.tjDecompressHeader3
        self.__decompress_header.argtypes = [
            c_void_p,
            POINTER(c_ubyte),
            c_ulong,
            POINTER(c_int),
            POINTER(c_int),
            POINTER(c_int),
            POINTER(c_int),
        ]
        self.__decompress_header.restype = c_int
        self.__decompress = turbo_jpeg.tjDecompress2
        self.__decompress.argtypes = [
            c_void_p,
            POINTER(c_ubyte),
            c_ulong,
            POINTER(c_ubyte),
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
        ]
        self.__decompress.restype = c_int
        self.__compress = turbo_jpeg.tjCompress2
        self.__compress.argtypes = [
            c_void_p,
            POINTER(c_ubyte),
            c_int,
            c_int,
            c_int,
            c_int,
            POINTER(c_void_p),
            POINTER(c_ulong),
            c_int,
            c_int,
            c_int,
        ]
        self.__compress.restype = c_int
        self.__free = turbo_jpeg.tjFree
        self.__free.argtypes = [c_void_p]
        self.__free.restype = None
        self.__get_error_str = turbo_jpeg.tjGetErrorStr
        self.__get_error_str.restype = c_char_p
        self.__get_error_str2 = getattr(turbo_jpeg, "tjGetErrorStr2", None)
        if self.__get_error_str2 is not None:
            self.__get_error_str2.argtypes = [c_void_p]
            self.__get_error_str2.restype = c_char_p
        self.__get_error_code = getattr(turbo_jpeg, "tjGetErrorCode", None)
        if self.__get_error_code is not None:
            self.__get_error_code.argtypes = [c_void_p]
            self.__get_error_code.restype = c_int
        self.__scaling_factors = []

        class ScalingFactor(Structure):
            _fields_ = ("num", c_int), ("denom", c_int)

        get_scaling_factors = turbo_jpeg.tjGetScalingFactors
        get_scaling_factors.argtypes = [POINTER(c_int)]
        get_scaling_factors.restype = POINTER(ScalingFactor)
        num_scaling_factors = c_int()
        scaling_factors = get_scaling_factors(byref(num_scaling_factors))
        for i in range(num_scaling_factors.value):
            self.__scaling_factors.append(
                (scaling_factors[i].num, scaling_factors[i].denom)
            )

    def decode_header(self, jpeg_buf):
        """decodes JPEG header and returns image properties as a tuple.
        e.g. (width, height, jpeg_subsample, jpeg_colorspace)
        """
        handle = self.__init_decompress()
        try:
            width = c_int()
            height = c_int()
            jpeg_subsample = c_int()
            jpeg_colorspace = c_int()
            jpeg_array = np.frombuffer(jpeg_buf, dtype=np.uint8)
            src_addr = self.__getaddr(jpeg_array)
            status = self.__decompress_header(
                handle,
                src_addr,
                jpeg_array.size,
                byref(width),
                byref(height),
                byref(jpeg_subsample),
                byref(jpeg_colorspace),
            )
            if status != 0:
                self.__report_error(handle)
            return (
                width.value,
                height.value,
                jpeg_subsample.value,
                jpeg_colorspace.value,
            )
        finally:
            self.__destroy(handle)

    def decode(self, jpeg_buf, pixel_format=TJPF_BGR, scaling_factor=None, flags=0):
        """decodes JPEG memory buffer to numpy array."""
        handle = self.__init_decompress()
        try:
            if (
                scaling_factor is not None
                and scaling_factor not in self.__scaling_factors
            ):
                raise ValueError(
                    "supported scaling factors are " + str(self.__scaling_factors)
                )
            pixel_size = [3, 3, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4]
            width = c_int()
            height = c_int()
            jpeg_subsample = c_int()
            jpeg_colorspace = c_int()
            jpeg_array = np.frombuffer(jpeg_buf, dtype=np.uint8)
            src_addr = self.__getaddr(jpeg_array)
            status = self.__decompress_header(
                handle,
                src_addr,
                jpeg_array.size,
                byref(width),
                byref(height),
                byref(jpeg_subsample),
                byref(jpeg_colorspace),
            )
            if status != 0:
                self.__report_error(handle)
            scaled_width = width.value
            scaled_height = height.value
            if scaling_factor is not None:

                def get_scaled_value(dim, num, denom):
                    return (dim * num + denom - 1) // denom

                scaled_width = get_scaled_value(
                    scaled_width, scaling_factor[0], scaling_factor[1]
                )
                scaled_height = get_scaled_value(
                    scaled_height, scaling_factor[0], scaling_factor[1]
                )
            img_array = np.empty(
                [scaled_height, scaled_width, pixel_size[pixel_format]], dtype=np.uint8
            )
            dest_addr = self.__getaddr(img_array)
            status = self.__decompress(
                handle,
                src_addr,
                jpeg_array.size,
                dest_addr,
                scaled_width,
                0,
                scaled_height,
                pixel_format,
                flags,
            )
            if status != 0:
                self.__report_error(handle)
            return img_array
        finally:
            self.__destroy(handle)

    def encode(
        self,
        img_array,
        quality=85,
        pixel_format=TJPF_BGR,
        jpeg_subsample=TJSAMP_422,
        flags=0,
    ):
        """encodes numpy array to JPEG memory buffer."""
        handle = self.__init_compress()
        try:
            jpeg_buf = c_void_p()
            jpeg_size = c_ulong()
            height, width, _ = img_array.shape
            src_addr = self.__getaddr(img_array)
            status = self.__compress(
                handle,
                src_addr,
                width,
                img_array.strides[0],
                height,
                pixel_format,
                byref(jpeg_buf),
                byref(jpeg_size),
                jpeg_subsample,
                quality,
                flags,
            )
            if status != 0:
                self.__report_error(handle)
            dest_buf = create_string_buffer(jpeg_size.value)
            memmove(dest_buf, jpeg_buf.value, jpeg_size.value)
            self.__free(jpeg_buf)
            return dest_buf.raw
        finally:
            self.__destroy(handle)

    def __report_error(self, handle):
        """reports error while error occurred"""
        if self.__get_error_code is not None:
            if self.__get_error_code(handle) == TJERR_WARNING:
                warnings.warn(self.__get_error_string(handle))
                return
        raise IOError(self.__get_error_string(handle))

    def __get_error_string(self, handle):
        """returns error string"""
        if self.__get_error_str2 is not None:
            return self.__get_error_str2(handle).decode()
        return self.__get_error_str().decode()

    def __find_turbojpeg(self):
        """returns default turbojpeg library path if possible"""
        lib_path = find_library("turbojpeg")
        if lib_path is not None:
            return lib_path
        for lib_path in DEFAULT_LIB_PATHS[platform.system()]:
            if os.path.exists(lib_path):
                return lib_path
        if platform.system() == "Linux" and "LD_LIBRARY_PATH" in os.environ:
            ld_library_path = os.environ["LD_LIBRARY_PATH"]
            for path in ld_library_path.split(":"):
                lib_path = os.path.join(path, "libturbojpeg.so.0")
                if os.path.exists(lib_path):
                    return lib_path
        raise RuntimeError(
            "Unable to locate turbojpeg library automatically. "
            "You may specify the turbojpeg library path manually.\n"
            "e.g. jpeg = TurboJPEG(lib_path)"
        )

    def __getaddr(self, nda):
        """returns the memory address for a given ndarray"""
        return cast(nda.__array_interface__["data"][0], POINTER(c_ubyte))


class Crc8DvbS2(object):
    """CRC-8/DVB-S2"""

    def __init__(self, initvalue=None):
        self._value = 0x00

    def process(self, data):
        crc = self._value
        for byte in data:
            crc = crc ^ byte
            for _ in range(0, 8):
                if crc & 0x80:
                    crc = (crc << 1) ^ 0xD5
                else:
                    crc = crc << 1
            crc &= 0xFF
        self._value = crc
        return self

    def final(self):
        crc = self._value
        crc ^= 0x00
        return crc

    @classmethod
    def calc(cls, data, initvalue=None, **kwargs):
        inst = cls(initvalue, **kwargs)
        inst.process(data)
        return inst.final()


class RobotAPI:
    port = None
    server_flag = False
    last_key = -1
    last_frame = np.ones((480, 640, 3), dtype=np.uint8)
    quality = 50
    manual_regim = 0
    manual_video = 1
    manual_speed = 150
    manual_angle = 0
    frame = np.ones((480, 640, 3), dtype=np.uint8)
    __joy_data = []
    __mouse_data = []
    small_frame = 0
    motor_left = 0
    motor_rigth = 0
    flag_serial = False
    flag_pyboard = False
    time_frame = time.time() + 1000
    __cap = []
    __num_active_cam = 0
    stop_frames = False
    quality = 20

    def __init__(
        self,
        flag_video=True,
        flag_keyboard=True,
        flag_serial=True,
        flag_pyboard=False,
        udp_stream=True,
        udp_turbo_stream=True,
        udp_event=True,
    ):
        self.flag_serial = flag_serial
        self.flag_pyboard = flag_pyboard

        atexit.register(self.cleanup)
        atexit.register(self.cleanup)

        self.flag_video = flag_video
        if self.flag_video == True:
            self.context = zmq.Context(1)
            self.socket = self.context.socket(zmq.REP)
            self.socket.setsockopt(zmq.SNDTIMEO, 3000)
            self.socket.setsockopt(zmq.RCVTIMEO, 3000)

            self.socket.bind("tcp://*:5555")
            cv2.putText(
                self.last_frame,
                str("Starting camera..."),
                (self.last_frame.shape[1] // 8, self.last_frame.shape[0] // 2),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                2,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                self.frame,
                str("Start..."),
                (self.last_frame.shape[1] // 3, self.last_frame.shape[0] // 2),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                2,
                (255, 255, 255),
                2,
            )

            self.server_flag = True
            self.my_thread_video = threading.Thread(target=self.__send_frame)
            self.my_thread_video.daemon = True

            self.my_thread_video.start()

            self.manual_video = 1

        if flag_keyboard:
            self.context2 = zmq.Context(1)
            self.socket2 = self.context2.socket(zmq.REP)

            self.socket2.bind("tcp://*:5559")
            self.server_keyboard = True
            self.my_thread = threading.Thread(target=self.__recive_key)
            self.my_thread.daemon = True
            self.my_thread.start()

        if udp_stream:
            self.my_thread_udp = threading.Thread(target=self.__work_udp)
            self.my_thread_udp.daemon = True
            self.my_thread_udp.start()

        if udp_turbo_stream:
            self.my_thread_turbo_udp = threading.Thread(target=self.__work_turbo_udp)
            self.my_thread_turbo_udp.daemon = True
            self.my_thread_turbo_udp.start()

        if udp_event:
            self.my_thread_udp_event = threading.Thread(target=self.__work_udp_event)
            self.my_thread_udp_event.daemon = True
            self.my_thread_udp_event.start()

        self.my_thread_f = threading.Thread(target=self.__work_f)
        self.my_thread_f.daemon = True
        self.my_thread_f.start()
        print("|start_api")

    def end_work(self):
        # self.cap.release()
        if self.flag_video:
            for i in self.__cap:
                if i is not None:
                    i.release()
        self.stop_frames = True
        time.sleep(0.3)
        self.frame = np.array([[10, 10], [10, 10]], dtype=np.uint8)
        time.sleep(0.3)
        print("|STOPED api")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_work()

    def cleanup(self):
        self.end_work()

    def joy(self):
        j = self.__joy_data.copy()
        self.__joy_data = []
        return j

    def mouse(self):
        m = self.__mouse_data.copy()
        self.__mouse_data = []
        return m

    def __work_udp(self):

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setblocking(0)
        host = ""
        port = 5001
        server_address = (host, port)

        sock.bind(server_address)

        while 1:
            try:
                data = b""
                data, address = sock.recvfrom(1)

                data = data.decode("utf-8")
                # print(address)
                if data == "g":
                    encode_param = [int(cv2.IMWRITE_JPEG_LUMA_QUALITY), self.quality]
                    _, buffer = cv2.imencode(".jpg", self.last_frame, encode_param)

                    if len(buffer) > 65507:
                        print(
                            "The message is too large to be sent within a single UDP datagram. We do not handle splitting the message in multiple datagrams"
                        )
                        sock.sendto("FAIL".encode("utf-8"), address)
                        continue
                    sock.sendto(buffer, address)
            except:
                time.sleep(0.01)
                pass

        pass

    def __work_turbo_udp(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            # Bind the socket to the port
            host = ""
            port = 5002
            server_address = (host, port)

            sock.bind(server_address)

            try:
                jpeg = TurboJPEG()
            except:
                pass
            jpeg_encode_func = (
                lambda img, jpeg_quality=self.quality, jpeg=jpeg: turbo_encode_image(
                    img, jpeg, jpeg_quality
                )
            )
            while 1:
                try:
                    # print("start TURBO frame", time.time())
                    data, address = sock.recvfrom(1)
                    data = data.decode("utf-8")

                    if data == "g":

                        buffer = jpeg_encode_func(self.last_frame)

                        if len(buffer) > 65507:
                            print(
                                "The message is too large to be sent within a single UDP TURBO datagram. We do not handle splitting the message in multiple datagrams"
                            )
                            sock.sendto("FAIL".encode("utf-8"), address)
                            continue
                        sock.sendto(buffer, address)
                except:
                    pass
        except:
            pass

    def __work_udp_event(self):

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        host = ""
        port = 5003
        server_address = (host, port)
        sock.bind(server_address)

        while 1:
            data = b""
            data, address = sock.recvfrom(65507)

            message = data.decode("utf-8")
            if len(message) > 0:
                if message.find("m") > -1:
                    self.__mouse_data = message.split(",")[1:]

                elif message.find("j") > -1:
                    self.__joy_data = message.split(",")[1:]
                else:
                    self.last_key = int(message.lstrip())

    def __recive_key(self):
        while True:
            try:
                message = ""
                try:
                    message = self.socket2.recv_string()
                except:
                    pass

                if len(message) > 0:
                    if message.find("m") > -1:
                        self.__mouse_data = message.split(",")[1:]
                    elif message.find("j") > -1:
                        self.__joy_data = message.split(",")[1:]
                    else:
                        self.last_key = int(message.lstrip())

                    try:
                        self.socket2.send_string("|")
                    except:
                        pass
                else:
                    time.sleep(0.001)
            except:
                time.sleep(0.001)

    def __work_f(self):
        self.stop_frames = False
        while True:

            if self.stop_frames == False and self.flag_video:
                if len(self.__cap) > 0:
                    if self.__cap[self.__num_active_cam] is not None:
                        ret, frame = self.__cap[self.__num_active_cam].read()

                        if ret is not None:
                            self.frame = frame
                            self.time_frame = time.time()
                        else:
                            self.stop_frames = True
                    else:
                        time.sleep(0.001)
                else:
                    time.sleep(0.001)

            else:
                time.sleep(0.001)

    def get_key(self):
        l = self.last_key
        self.last_key = -1
        return l

    def __send_frame(self):
        time1 = time.time()
        md = 0
        frame = 0

        while True:
            if self.last_frame is not None:
                if self.server_flag == True:
                    message = ""
                    try:
                        message = self.socket.recv_string()
                    except:
                        pass

                    if message != "":
                        try:
                            if time1 < self.time_frame:
                                self.encode_param = [
                                    int(cv2.IMWRITE_JPEG_LUMA_QUALITY),
                                    self.quality,
                                ]
                                _, frame = cv2.imencode(
                                    ".jpg", self.last_frame, self.encode_param
                                )
                                time1 = time.time()

                                md = dict(
                                    # arrayname="jpg",
                                    dtype=str(frame.dtype),
                                    shape=frame.shape,
                                )
                            self.socket.send_json(md, zmq.SNDMORE)
                            self.socket.send(frame, 0)

                        except:
                            pass
                    else:
                        time.sleep(0.001)
                else:
                    time.sleep(0.001)
            else:
                time.sleep(0.001)

    def set_frame(self, frame, quality=30):
        self.quality = quality
        self.last_frame = frame

    def _fromInt16(self, value):
        return struct.unpack("<BB", struct.pack("@h", value))

    def _fromInt32(self, value):
        return struct.unpack("<BBBB", struct.pack("@i", value))

    def text_to_frame(self, frame, text, x, y, font_color=(255, 255, 255), font_size=2):
        cv2.putText(
            frame,
            str(text),
            (x, y),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            1,
            font_color,
            font_size,
        )
        return frame

    def distance_between_points(self, x1, y1, x2, y2):
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def map(self, x, in_min, in_max, out_min, out_max):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def median(self, lst):
        if len(lst) > 0:
            quotient, remainder = divmod(len(lst), 2)
            if remainder:
                return sorted(lst)[quotient]
            return sum(sorted(lst)[quotient - 1 : quotient + 1]) / 2.0

    def constrain(self, x, out_min, out_max):
        if x < out_min:
            return out_min
        elif out_max < x:
            return out_max
        else:
            return x
