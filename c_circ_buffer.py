"""
Circular buffer class
"""

# -------------------------
# Imports
# -------------------------

import numpy as np

# -------------------------
# Class def
# -------------------------

class Circ_buffer:
    """
    Circular buffer class
    """

    def __init__(self, buffer_size = 16, delay_size = 1):
        """
        :param buffer_size:
            type: int
            desc: buffer size, in terms of log2( total buffer size ).
                    actual buffer size = 2^(buffer_size)
        :param delay_size:
            type: int
            desc: delay amount.
        """

        # set buffer size and create buffer
        self.buffer_size = 2**buffer_size
        self.buffer = np.zeros(self.buffer_size)

        # set wire mask
        self.wrapmask = self.buffer_size - 1

        # initialize the write index
        self.write_index = 0

        # set delay and initialize read
        self.set_delay(delay_size)

    # end constructor


    def read_value(self, delay_size=None):
        """
        Read a value from the circular buffer
        Optional - user can specify delay to read from

        :param delay_size:
            type: int
            desc: OPTIONAL. if specified, then the function reads data with the given delay

        :return:
            value read from the buffer
        """

        if delay_size is None:
            return self.buffer[self.read_index]
        else:
            return self.buffer[ ( self.write_index - delay_size ) & self.wrapmask ]

    # end function read_value()


    def write_value(self, value):
        """
        Write a value to the current time position in the buffer
        Increment time

        :param value:
            type: arbitrary
            desc: value to write

        :return:
            nothing
        """
        self.buffer[self.write_index] = value
        # update time
        self._increment_time()
    # end function write_value()


    def _increment_time(self):
        """
        increment time by 1
        :return:
            nothing
        """
        self.write_index += 1
        self.write_index = self.write_index & self.wrapmask
        self.read_index += 1
        self.read_index = self.read_index & self.wrapmask
    # end function _increment_time()


    def set_delay(self, delay_size):
        """
        Sets delay time
        :param delay_size:
            type: int
            desc: amount of delay, specified in # of indices
        :return:
            nothing
        """
        self.delay_size = delay_size
        self.read_index = (self.write_index - self.delay_size) & self.wrapmask
    # end function set_delay

# end class Circ_buffer


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # debugging

    mybuff = Circ_buffer( buffer_size = 3, delay_size = 2 )

    print(mybuff.buffer_size)
    print(mybuff.buffer)

    mybuff.write_value( 100 )
    print(mybuff.buffer)

    for ii in range( mybuff.buffer_size-1 ):
        mybuff.write_value(ii)

    print(mybuff.buffer)

    print(mybuff.write_index)
    print(mybuff.read_value())

    mybuff.set_delay(4)
    print(mybuff.read_value())

    print(mybuff.read_value(7))