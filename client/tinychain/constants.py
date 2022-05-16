import logging

import math


e = math.e


DEFAULT_PORT = 8702
ENCODING = "utf-8"


# TODO: is there a better place to define this?
def debug(msg_maybe_lambda):
    """
    Write the given message to the debug log.
    Pass `lambda: msg` in case the message is computationally intensive to format.
    """

    if logging.getLogger().isEnabledFor(logging.DEBUG):
        if callable(msg_maybe_lambda):
            logging.debug(msg_maybe_lambda())
        else:
            logging.debug(msg_maybe_lambda)
