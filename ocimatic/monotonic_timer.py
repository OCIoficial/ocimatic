from ctypes import Structure, c_long, CDLL, c_int, POINTER, byref
from ctypes.util import find_library
CLOCK_MONOTONIC = 1


class timespec(Structure):
    _fields_ = [
        ('tv_sec', c_long),
        ('tv_nsec', c_long)
    ]


librt_filename = find_library('rt')
if not librt_filename:
    # On Debian Lenny (Python 2.5.2), find_library() is unable
    # to locate /lib/librt.so.1
    librt_filename = 'librt.so.1'
librt = CDLL(librt_filename)
_clock_gettime = librt.clock_gettime
_clock_gettime.argtypes = (c_int, POINTER(timespec))


def monotonic_time():
    t = timespec()
    _clock_gettime(CLOCK_MONOTONIC, byref(t))
    return t.tv_sec + t.tv_nsec / 1e9
