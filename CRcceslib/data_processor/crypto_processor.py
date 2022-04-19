import numpy as np
from sklearn import preprocessing


class ProcessorUtility:
    def __init__(self, verbose):
        self.verbose = verbose

    def get_train_test(self, crypto_df, validation_pct=0.05):
        """"""
        #
        timestamp = crypto_df.index.values
        last_5pct = timestamp[-int(validation_pct * len(timestamp))]  # Last 5 percent timestamp
        val_crypto_df = crypto_df[(crypto_df.index >= last_5pct)]
        train_crypto_df = crypto_df[(crypto_df.index < last_5pct)]
        return train_crypto_df, val_crypto_df

    def split_df(self, df_values, window_len):
        """"""
        out_data = None
        diff_data = None
        for times_try in range(len(df_values)):
            rest = len(df_values) - times_try
            data = df_values[0:rest]
            try:
                out_data = np.array(np.split(data, window_len))
                df_shape = out_data.shape
                out_data = out_data.reshape(df_shape[1], df_shape[0], df_shape[2])
                diff_data = df_values[rest:len(df_values)]
                break
            except ValueError:
                pass

        return out_data, diff_data

    def normalize_df(self, df, features, target_name):
        min_max_scaler = preprocessing.RobustScaler()
        x_val = min_max_scaler.fit_transform(df[features])
        y_val = df[target_name].values.reshape((len(df), 1))
        out_val = np.append(x_val, y_val, axis=1)
        return out_val

    def dataset_generator_lstm(self, df_values, columns_len, look_back=100):
        dataX, dataY = [], []
        for i in range(len(df_values) - look_back):
            window_size_x = df_values[i:(i + look_back), 0:(columns_len-1)]
            dataX.append(window_size_x)
            dataY.append(df_values[i + look_back, (columns_len-1)])
        return np.array(dataX), np.array(dataY).reshape((len(dataY), 1))


class ProcessorSequence(ProcessorUtility):
    def __init__(self, verbose=True):
        super().__init__(verbose)

    def process(self, crypto_df, target_name, sequence_length, validation_pct=0.05):
        """
        Reformat data to fit model.
        Note: Always the colum we want to predict must be in the last position of the data set
        """

        columns_len = len(crypto_df.columns)
        train_df, validation_df = self.get_train_test(crypto_df, validation_pct=validation_pct)
        if self.verbose:
            print(f"Train shape: {train_df.shape}, Validation shape: {validation_df.shape}")

        features = [col_name for col_name in train_df.columns if col_name != target_name]

        """ Normalize parameters columns """
        train_df_values = self.normalize_df(train_df, features, target_name)
        validation_df_values = self.normalize_df(validation_df, features, target_name)

        """ Generate dataset for lstm crypto model"""
        train_x, train_y = self.dataset_generator_lstm(train_df_values, columns_len, look_back=sequence_length)
        test_x, test_y = self.dataset_generator_lstm(validation_df_values, columns_len, look_back=sequence_length)

        if self.verbose:
            print(f"train_x shape: {train_x.shape}, train_y shape: {train_y.shape}")
            print(f"test_x shape: {test_x.shape}, test_y shape: {test_y.shape}")

        return train_x, train_y, test_x, test_y

    # def process(self, crypto_df, target_name, sequence_length, validation_pct=0.05):
    #     """Reformat data to fit model"""
    #
    #     train_df, validation_df = self.get_train_test(crypto_df, validation_pct=validation_pct)
    #     if self.verbose:
    #         print(f"Train shape: {train_df.shape}, Validation shape: {validation_df.shape}")
    #
    #     features = [col_name for col_name in train_df.columns if col_name != target_name]
    #
    #     """ Normalize parameters columns """
    #     train_df_val = self.normalize_df(train_df, features)
    #     validation_df_val = self.normalize_df(validation_df, features)
    #
    #     train_x, train_diff_x = self.split_df(train_df_val, sequence_length)
    #     train_y, train_diff_y = self.split_df(np.expand_dims(train_df[target_name].values, axis=1), sequence_length)
    #
    #     if train_diff_x.shape[0] == 0:
    #         del train_diff_x, train_diff_y
    #         if self.verbose:
    #             print("No data remaining in train chunks!")
    #     else:
    #         if self.verbose:
    #             print(f"train_x shape: {train_x.shape},  train_diff_x shape: {train_diff_x.shape}")
    #
    #     test_x, test_diff_x = self.split_df(validation_df_val, sequence_length)
    #     test_y, test_diff_y = self.split_df(np.expand_dims(validation_df[target_name].values, axis=1), sequence_length)
    #     if test_diff_x.shape[0] == 0:
    #         del test_diff_x
    #         if self.verbose:
    #             print("No data remaining in validation chunks!")
    #     else:
    #         if self.verbose:
    #             print(f"test_x shape: {test_x.shape},  test_diff_x shape: {test_diff_x.shape}")
    #
    #     return train_x, train_y, test_x, test_y
