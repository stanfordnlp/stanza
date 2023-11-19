"""
A couple more specific FileNotFoundError exceptions

The idea being, the caller can catch it and report a more useful error resolution
"""

import errno

class ForwardCharlmNotFoundError(FileNotFoundError):
    def __init__(self, msg, filename):
        super().__init__(errno.ENOENT, msg, filename)

class BackwardCharlmNotFoundError(FileNotFoundError):
    def __init__(self, msg, filename):
        super().__init__(errno.ENOENT, msg, filename)
