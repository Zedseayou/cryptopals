import binascii as ba
from typing import List

import numpy as np
import pandas as pd
import string
import re
import itertools
from math import sqrt


def hex_to_b64(hex_string: str) -> str:
    """Convert a hex string representation to a bytes object in base 64"""
    hex_bytes: bytes = ba.a2b_hex(hex_string)
    b64_string: str = ba.b2a_base64(hex_bytes, newline=False)
    return b64_string


def byte_to_str(bytes_obj: bytes) -> str:
    """Convert bytes object to string, removing header and trailing quote"""
    byte_string: str = str(bytes_obj)
    return byte_string.rstrip("'").lstrip("b'")


def fixed_xor(bytes1: bytes, bytes2: bytes) -> bytes:
    """Bitwise XOR two hex string representations together and return the bytes object"""
    assert len(bytes1) == len(bytes2), "Input lengths are not equal"
    xor_int = np.array(list(bytes1)) ^ np.array(list(bytes2))
    xor_bytes: bytes = bytes(list(xor_int))
    return xor_bytes


def onebyte_xor(bytes_obj: bytes, one_byte: int) -> bytes:
    """Bitwise XOR a bytes object representation against a single byte as integer and return the bytes object"""
    assert isinstance(bytes_obj, bytes), "`bytes_obj` is not str type"
    assert isinstance(one_byte, int), "`one_byte` is not int type"
    assert one_byte in range(256), "`one_byte` is not a single byte"
    xor_int = [int(one_byte ^ i) for i in bytes_obj]
    xor_bytes = bytes(xor_int)
    return xor_bytes


def rfreq_letter(string_in: str) -> List[float]:
    """Get the relative frequency of each letter in a string as a list of doubles"""
    assert isinstance(string_in, str), "`string` is not str type"
    letters_only = re.sub("[^A-z]", "", string_in)
    uppercase_in = letters_only.upper()
    rfreqs = [uppercase_in.count(letter) / len(letters_only) for letter in string.ascii_uppercase]
    return rfreqs


def rfreq_byte(bytes_in: bytes) -> List[float]:
    """Get the relative frequency of each byte in a bytes object as a list of doubles"""
    assert isinstance(bytes_in, bytes), "`bytes_in` is not bytes type"
    rfreqs = [bytes_in.count(i) / 256 for i in range(256)]
    return rfreqs


def ascii_hex(bytes_obj: bytes) -> float:
    """Get the fraction of ASCII range bytes in a bytes object as a double"""
    assert isinstance(bytes_obj, bytes), "`byte_hex` is not bytes type"
    num_ascii: int = sum([bytes_obj[i] in range(32, 127) for i in range(len(bytes_obj))])
    return num_ascii / len(bytes_obj)


def rmse(list1, list2) -> float:
    """Calculate the root mean squared error of two lists of numbers"""
    assert len(list1) == len(list2), "`list1` is not same length as `list2`"
    arr1, arr2 = (np.array(num_list) for num_list in (list1, list2))
    rms: float = np.mean((arr1 - arr2) ** 2)
    return rms


def BC(dist1, dist2) -> float:
    """
    Calculate the Bhattarcharya distance between two discrete distributions. Distributions should be supplied as 1D
    NumPy arrays of equal length, or else as lists which will be converted to NumPy arrays.
    :param dist1: First distribution X, list of values p(X) = x
    :param dist2: Second distribution Y, list of values p(Y) = y
    :return: Bhattarcharya coefficient between 0 and 1
    """
    assert len(dist1) == len(dist2), "`dist1` is not the same length as `dist2`"
    arr1, arr2 = (np.array(num_list) for num_list in (dist1, dist2))
    bcoef: float = sum(np.sqrt(arr1 * arr2))
    return bcoef


eng_freq = dict(
    A=8.167, B=1.492, C=2.782, D=4.253, E=12.702, F=2.228, G=2.015, H=6.094, I=6.966, J=0.153, K=0.772,
    L=4.025, M=2.406, N=6.749, O=7.507, P=1.929, Q=0.095, R=5.987, S=6.327, T=9.056, U=2.758, V=0.978,
    W=2.360, X=0.150, Y=1.974, Z=0.074
)


def freq_rmse(string_in: str, freq_compare=np.array(list(eng_freq.values())) / 100) -> float:
    """Calculate the rmse similarity between a string's letter frequency and English"""
    string_rfreqs: List[float] = rfreq_letter(string_in)
    rms: float = rmse(string_rfreqs, freq_compare)
    return rms


def freq_ols(string_in: str, freq_compare=list(eng_freq.values())) -> float:
    """Calclate the least squares similarity between a string's letter frequency and English"""
    string_rfreqs: List[float] = rfreq_letter(string_in)
    ols: float = sum((np.array(string_rfreqs) - np.array(freq_compare)) ** 2)
    return ols


def freq_BC(string_in: str, freq_compare=np.array(list(eng_freq.values())) / 100) -> float:
    """Calculate the Bhattachandrya coefficient between a string's letter frequency and English"""
    string_rfreqs: List[float] = rfreq_letter(string_in)
    bcoef: float = BC(string_rfreqs, freq_compare)
    return bcoef


def best_eng_onebyte(bytes_obj: bytes) -> pd.Series:
    """
    Return the output from the best single-byte XOR for a string.
    :param bytes_obj:
    :return: A series containing the XOR integer, character, output, fraction of ASCII characters in the output, and
    the Bhattachandrya coefficient for the XOR output letter frequency as compared to English.
    """

    df = pd.DataFrame({"xor_int": np.array(range(32, 127))})
    df = df.assign(xor_chr=df.xor_int.apply(lambda x: byte_to_str(bytes([x]))))
    df = df.assign(xored_bytes=df.xor_int.apply(lambda x: onebyte_xor(bytes_obj, x)))
    df = df.assign(ascii_frac=df.xored_bytes.apply(ascii_hex))
    df = df.assign(xor_bcoef=df.xored_bytes.apply(lambda x: freq_BC(byte_to_str(x))))
    return df.iloc[df.xor_bcoef.idxmax()]


def flatten(nested_list: List) -> List:
    """
    Flattens a nested list to a single list
    :param nested_list:
    :return: List with one layer of hierarchy removed
    """
    flat_list = [item for sublist in nested_list for item in sublist]
    return flat_list


def repeat_xor(string_in: str, key: str) -> bytes:
    """
    Encode a string using repeating-key XOR.
    :param string_in: A plaintext ASCII string to be encrypted.
    :param key: A plaintext ASCII key to encode the string with.
    :return: A bytes object containing the encoded bytes, in hex representation
    """
    string_ints: List[int] = [i for i in bytes(string_in, "ASCII")]
    key_ints: List[int] = [i for i in bytes(key, "ASCII")]
    repetitions: int = len(string_ints) // len(key_ints)
    remainder: int = len(string_ints) % len(key_ints)
    rep_key_ints: List[int] = flatten(list(itertools.repeat(key_ints, times=repetitions))) + key_ints[0:remainder]
    assert len(string_ints) == len(rep_key_ints), "Arrays are not equal length (internal bug)"
    xor_int: List[int] = [int(i[0] ^ i[1]) for i in zip(string_ints, rep_key_ints)]
    xor_hex = ba.b2a_hex(bytes(xor_int))
    return xor_hex


def int_to_binary(int_in: int) -> str:
    """
    Convert an integer representing one byte into its binary representation in string form, including leading zeroes.
    :int_in: Integer between 0 and 255 to convert to binary representation.
    :return: String of length 8 containing binary representation of input integer.
    """
    bin_out: str = bin(int_in)[2:].zfill(8)
    return bin_out


def str_to_binary(string_in: str) -> str:
    """
    Convert a character string to its binary representation in string form, including leading zeroes.
    :param string_in:
    :return:
    """
    bin_chars = [int_to_binary(ord(c)) for c in string_in]
    bin_str: str = "".join(bin_chars)
    return bin_str


def hamming_dist(bytes_in1, bytes_in2) -> int:
    """
    Compute the Hamming distance between two objects, which is the number of differing bits.
    :param bytes_in1: Either ASCII string or list of integers, representing one byte each.
    :param bytes_in2:
    :return: Integer number of differing bits between the two objects
    """
    assert len(bytes_in2) == len(bytes_in1), "Input lengths are not equal"
    if isinstance(bytes_in1, str):
        bin1: List[int] = [int(d) for d in str_to_binary(bytes_in1)]
    else:
        bin1: List[int] = bytes_in1
    if isinstance(bytes_in2, str):
        bin2: List[int] = [int(d) for d in str_to_binary(bytes_in2)]
    else:
        bin2: List[int] = bytes_in2
    dist: int = sum(np.array(bin1) ^ np.array(bin2))
    return dist


def norm_keysize(bytes_in, block_size: int) -> float:
    """
    Calculates normalised Hamming distance as step in finding the best keysize to decode a ciphertext.
    Takes first two pairs of byte blocks and calculates their Hamming distance, averages and normalises by block length
    :param block_size: Number of bytes to include in each block
    :param bytes_in: List of bytes to assess keysize suitability on
    # :param pairs: Number of pairs of blocks to average
    :return: Normalised Hamming distance
    """
    pair1 = (bytes_in[block_size * 0:block_size * 1], bytes_in[block_size * 1:block_size * 2])
    pair2 = (bytes_in[block_size * 2:block_size * 3], bytes_in[block_size * 3:block_size * 4])
    norm_keysize: float = (hamming_dist(pair1[0], pair1[1]) + hamming_dist(pair2[0], pair2[1])) / block_size
    return norm_keysize
