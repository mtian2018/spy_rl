import os
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO)
# console_format = logging.Formatter('%(levelname)s: %(name)s: %(message)s')
# console_handler.setFormatter(console_format)
# logger.addHandler(console_handler)

if not os.path.isfile('spy.h5'):
    try:
        df = pd.read_csv('SPY.csv',
                         index_col=0,
                         parse_dates=True,
                         infer_datetime_format=True)
    except FileNotFoundError as e:
        logging.warn(e)
    else:
        df.to_hdf('spy.h5', key='first')
else:
    store = pd.HDFStore('spy.h5', mode='r')
    logging.info(store.info())
    store.close()

df = pd.read_hdf('spy.h5', key='processed')
df = df.round(decimals=2)
df.to_hdf('spy.h5', key='rounded')

df.drop(columns=['Adj Close'], inplace=True)
print(df.head())
df.to_hdf('spy.ht', key='drop_adj')