import logging
import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt


logger = logging.getLogger('__name__')
# c_handler = logging.StreamHandler()\
# c_handler.setLevel(logging.INFO)
# c_handler
"""
make, select random start time, sub df
reset, select another date?
step, next bar
state, reward, action
state, ohlc, ma, etc,
reward, price diff
action, buy sell hold
stop loss
"""

BATCH_SIZE = 200


class Quotes:
    """simulate gym with stock quotes"""

    def __init__(self):
        self._df = pd.read_hdf('spy.h5', key='rounded')
        self._batch = pd.DataFrame()

        self._done = True
        self._day = 0
        self._position = 0   # long short
        self._holding = 0    # entry price
        self._virtual_gain = 0
        self._realized_gain = 0
        self._total_gain = 0

    def reset(self):
        self._done = False
        self._day = 0
        self._position = 0
        self._holding = 0
        self._virtual_gain = 0
        self._realized_gain = 0
        self._total_gain = 0

        start = np.random.randint(0, len(self._df) - BATCH_SIZE + 1)
        # logger.info(f"starting day is {start_day}")
        # print(self._df.iloc[[start]])
        self._batch = self._df.iloc[start:start + BATCH_SIZE].values

        return self._batch[self._day]

    def step(self, action): # 0 hold, 1 buy, -1 sell
        """calculate positions and move to next day"""

        assert(not self._done), "too many steps"

        self._virtual_gain = self._position * (self._df.iat[self._day, 3] - self._holding)
        self._total_gain += self._virtual_gain

        act_pos = action * self._position
        if action == 0 or act_pos == 1:  # hold, buy buy, sell sell
            pass
        elif act_pos == 0:
            self._trade_open(action)
        elif act_pos == -1:
            self._trade_close()
        else:
            logging.warn("action position combination gone wild")

        assert(self._position in [-1, 0, 1]), "position out of range"

        self._day += 1
        self.is_done()

        result = (self._batch[self._day], self._total_gain, self._done, None)
        return result

    def _trade_open(self, action):
        self._position += action
        self._holding = self._df.iat[self._day, 3]

    def _trade_close(self):
        self._holding = 0
        self._position = 0
        self._realized_gain += self._virtual_gain

    def is_done(self):
        if self._day >= BATCH_SIZE - 1:
            self._done = True
        # elif self._gain / self._holding < -.2 or
        #     self._gain / self._holding > 1.0
        else:
            pass

    def enf_pram(self):
        result = (self._df.shape[1], 3)
        return result
