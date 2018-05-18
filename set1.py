import binascii as ba
import numpy as np
import string
import re
from math import sqrt


def hex_to_b64(hex_string: str) -> str:
    """Convert a hex string representation to a bytes object in base 64"""
    hex_bytes: bytes = ba.a2b_hex(hex_string)
    b64_string: str = ba.b2a_base64(hex_bytes, newline=False)
    return b64_string


def hex_to_arr(hex_string: str):
    """Convert a bytes object to a NumPy array of integers, one per byte"""
    int_array = np.array([i for i in ba.a2b_hex(hex_string)])
    return int_array


def byte_to_str(bytes_obj: bytes) -> str:
    """Convert bytes object to string, removing header and trailing quote"""
    byte_string: str = str(bytes_obj)
    return byte_string.rstrip("'").lstrip("b'")


def fixed_xor(hex_string1: str, hex_string2: str):
    """Bitwise XOR two hex string representations together and return the bytes object"""
    assert len(hex_string1) == len(hex_string2), "`hex_string1` and `hex_string2` are not equal length"
    bytes1, bytes2 = (ba.a2b_hex(str) for str in [hex_string1, hex_string2])
    xor_int = [int(i[0] ^ i[1]) for i in zip(bytes1, bytes2)]
    xor_hex = ba.b2a_hex(bytes(xor_int))
    return xor_hex


def onebyte_xor(hexstr, char_int):
    assert isinstance(hexstr, str), "`hexstr` is not str type"
    assert isinstance(char_int, int), "`char_int` is not int type"
    assert char_int in range(256), "`char_int` is not a single byte"
    str_byte = ba.a2b_hex(hexstr)
    xor_int = [int(char_int ^ i) for i in str_byte]
    xor_hex = bytes(xor_int)
    return xor_hex


def ascii_hex(byte_hex):
    assert isinstance(byte_hex, bytes), "`byte_hex` is not bytes type"
    num_ascii = sum([byte_hex[i] in range(32, 127) for i in range(len(byte_hex))])
    return num_ascii / len(byte_hex)


def letter_freq(string_in):
    assert isinstance(string_in, str), "`string` is not str type"
    uppercase_in = string_in.upper()
    counts = [len(re.findall(letter, uppercase_in)) for letter in string.ascii_uppercase]
    freqs = [ct / len(string_in) for ct in counts]
    return freqs


def rmse(list1, list2):
    assert len(list1) == len(list2), "`list1` is not same length as `list2`"
    sqe = [(x[0] - x[1]) ** 2 for x in zip(list1, list2)]
    rms = sqrt(sum(sqe) / len(sqe))
    return rms


eng_freq = dict(
    A=8.167, B=1.492, C=2.782, D=4.253, E=12.702, F=2.228, G=2.015, H=6.094, I=6.966, J=0.153, K=0.772,
    L=4.025, M=2.406, N=6.749, O=7.507, P=1.929, Q=0.095, R=5.987, S=6.327, T=9.056, U=2.758, V=0.978,
    W=2.360, X=0.150, Y=1.974, Z=0.074
)


def freq_rmse(string_in, freq_compare=list(eng_freq.values())):
    string_freqs = letter_freq(string_in)
    rms = rmse(string_freqs, freq_compare)
    return rms
