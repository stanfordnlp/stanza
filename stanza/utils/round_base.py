def round_base(num, base=10):
    """
    Rounding a number to its nearest multiple of the base. round_base(49.2, base=50) = 50.
    """
    return base * round(num / base)