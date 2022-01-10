import pandas as pd
import numpy as np


def get_train_test(df_path, validation_pct=0.05):
    crypto_df = pd.read_csv(df_path, index_col=0)
    timestamp = crypto_df.index.values
    last_5pct = timestamp[-int(validation_pct * len(timestamp))]  # Last 5 percent timestamp
    val_crypto_df = crypto_df[(crypto_df.index >= last_5pct)]
    train_crypto_df = crypto_df[(crypto_df.index < last_5pct)]
    return train_crypto_df, val_crypto_df


def split_target(df, window_len):
    out_data = None
    diff_data = None
    for times_try in range(len(df)):
        rest = len(df) - times_try
        data = df.values[0:rest]
        try:
            out_data = np.array(np.split(data, window_len))
            df_shape = out_data.shape
            out_data = out_data.reshape(df_shape[1], df_shape[0], df_shape[2])
            diff_data = train_df.values[rest:len(df)]
            break
        except ValueError:
            pass

    return out_data, diff_data


if __name__ == "__main__":
    SEQ_LEN = 120  # 300
    FUTURE_PERIOD_PREDICT = 15
    RATIO_TO_PREDICT = 'BTC-USD'
    PATH = "/home/titoare/Documents/ds/g_research_cr/input/j_data/btc.csv"
    train_df, val_df = get_train_test(PATH, validation_pct=0.05)
    print(f"Train shape: {train_df.shape}, Validation shape: {val_df.shape}")

    output_data_train, diff_data_train = split_target(train_df, SEQ_LEN)
    print("Train:")
    print(output_data_train.shape)
    print(diff_data_train.shape)

    output_data_val, diff_data_val = split_target(val_df, SEQ_LEN)
    print("\nValidation:")
    print(output_data_val.shape)
    print(diff_data_val.shape)
