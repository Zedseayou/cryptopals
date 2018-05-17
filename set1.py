import binascii as ba


def hex2b64(hexstr):
    hex_bytes =  ba.a2b_hex(hexstr)
    b64str = ba.b2a_base64(hex_bytes, newline = False)
    return b64str


def bytes2str(bytes_obj):
    byte_string = str(bytes_obj)
    return byte_string.rstrip("'").lstrip("b'")


def fixed_xor(hexstr1, hexstr2):
    assert len(hexstr1) == len(hexstr2), "`hexstr1` and `hexstr2` are not equal length"
    bytes1, bytes2 = (ba.a2b_hex(str) for str in [hexstr1, hexstr2])
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


def score_hex(byte_hex):
    assert isinstance(byte_hex, bytes), "`byte_hex` is not bytes type"
    num_ascii = sum([byte_hex[i] in range(32, 127) for i in range(len(byte_hex))])
    return num_ascii / len(byte_hex)
