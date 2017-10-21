"""
Misc utilities involving printing
"""

from datetime import datetime

def print_with_timestamp(message):
    timestamp = "[" + str(datetime.now()) + "]"
    print(str(timestamp) + " " + message)
