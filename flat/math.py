# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************


def ceil_divide(x: int, y: int) -> int:
    return (x + y - 1) // y


def divide_if_divisible(dividend: int, divisor: int, msg: str = "") -> int:
    assert dividend % divisor == 0, msg
    return dividend // divisor
