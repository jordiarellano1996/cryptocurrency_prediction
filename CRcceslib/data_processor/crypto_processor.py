import numpy as np


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


class ProcessorSequence(ProcessorUtility):
    def __init__(self, verbose=True):
        super().__init__(verbose)

    def process(self, crypto_df, target_name, sequence_length, validation_pct=0.05):
        """Reformat data to fit model"""

        train_df, validation_df = self.get_train_test(crypto_df, validation_pct=validation_pct)
        if self.verbose:
            print(f"Train shape: {train_df.shape}, Validation shape: {validation_df.shape}")

        features = [col_name for col_name in train_df.columns if col_name != target_name]

        train_x, train_diff_x = self.split_df(train_df[features].values, sequence_length)
        train_y, train_diff_y = self.split_df(np.expand_dims(train_df[target_name].values, axis=1), sequence_length)

        if train_diff_x.shape[0] == 0:
            del train_diff_x, train_diff_y
            if self.verbose:
                print("No data remaining in train chunks!")
        else:
            if self.verbose:
                print(f"train_diff_x shape: {train_diff_x.shape}")

        test_x, test_diff_x = self.split_df(validation_df[features].values, sequence_length)
        test_y, test_diff_y = self.split_df(np.expand_dims(validation_df[target_name].values, axis=1), sequence_length)
        if test_diff_x.shape[0] == 0:
            del test_diff_x
            if self.verbose:
                print("No data remaining in validation chunks!")
        else:
            if self.verbose:
                print(f"train_diff_x shape: {test_diff_x.shape}")

        return train_x, train_y, test_x, test_y
