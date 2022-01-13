import numpy as np


class DataTransformUtility:
    """"""

    def __init__(self, data_frame, verbose=True):
        self.__df = data_frame.copy()
        self.verbose = verbose

    def fill_timestamp_gaps(self, granularity=60):
        """"""
        self.__df = self.__df.reindex(range(self.__df.index[0], self.__df.index[-1] + granularity, granularity),
                                      method='pad')

    def add_vwap(self):
        """"""
        # price = (self.__df.low + self.__df.high + self.__df.close) / 3
        # self.__df["VWAP"] = (self.__df.volume * price) / self.__df.volume

    def window_price_fluctuation(self, window_minute):
        """Get close price increment/decrement in time window """
        self.__df["timestamp"] = self.__df.index
        self.__df.set_index("time", inplace=True)
        self.__df['p1'] = self.__df["close"].shift(freq='-1T')
        self.__df['p2'] = self.__df["close"].shift(freq=f'-{window_minute + 1}T')
        self.__df[f'return_{window_minute}min'] = np.log(self.__df.p2 / self.__df.p1)
        self.__df.drop(['p1', 'p2'], axis=1, inplace=True)
        self.__df.set_index("timestamp", inplace=True)

    def sort_by_index(self, ascending_in=True):
        self.__df.sort_index(ascending=ascending_in, inplace=True)

    def get_df(self):
        """"""
        return self.__df


class DTSequence(DataTransformUtility):
    """"""
    def __init__(self, data_frame):
        super().__init__(data_frame)

    def process(self, window_target_in, granularity_in):
        """In this case we are using a target window of 15 minute, and a data granularity of 60 seconds"""
        self.fill_timestamp_gaps(granularity=granularity_in)
        self.window_price_fluctuation(window_target_in)
        self.sort_by_index()
        df = self.get_df()
        df.drop(['low', 'high', 'open', 'time'], axis=1, inplace=True)
        return df
