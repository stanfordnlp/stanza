from contextlib import contextmanager
import argparse

@contextmanager
def argparse_full_errors():
    """
    A useful context window for printing out more information about an
    argparse error when it happens

    can run a test as follows:

    with argparse_full_errors():
       # do test stuff here
    """
    original = argparse.ArgumentParser.error
    def error_with_traceback(self, message):
        raise ValueError(f"argparse error: {message}\nprog: {self.prog}")
    argparse.ArgumentParser.error = error_with_traceback
    try:
        yield
    finally:
        argparse.ArgumentParser.error = original
