def find_nth(haystack, needle, n):
    """
    Returns the starting index of the nth occurrence of the substring 'needle' in the string 'haystack'.
    """
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start + len(needle))
        n -= 1
    return start


def round_base(num, base=10):
    """
    Rounding a number to its nearest multiple of the base. round_base(49.2, base=50) = 50.
    """
    return base * round(num / base)