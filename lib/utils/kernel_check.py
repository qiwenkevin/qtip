# lib/utils/kernel_check.py
def has_kernel(decode_mode, L, K, V, tlut_bits, td_x, td_y):
    if L != 16:
        return False
    if K < 2 or K > 4:
        return False
    if tlut_bits != 9:
        return False
    if td_x != 16 or td_y != 16:
        return False
    if decode_mode == 'quantlut_sym' and V == 2:
        return True
    if decode_mode == 'custom' and V == 1:
        return True
    return False
