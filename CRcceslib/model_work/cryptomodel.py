from CRcceslib.data_processor import crypto_processor, sequence_transform
import time
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, Input
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint


class Config:
    SEED = [2022, ]
    SEQ_LEN = 120  # 300
    FUTURE_PERIOD_PREDICT = 15
    RATIO_TO_PREDICT = 'BTC-USD'
    N_A = 128
    EPOCHS = 1
    BATCH_SIZE = 64
    TARGET_NAME = "return_15min"
    NAME = f"{SEQ_LEN}_SEQ_{FUTURE_PERIOD_PREDICT}_PRED_{int(time.time())}"
    TRAIN_MODEL = False


def create_model(tr_x, n_a):
    """
    Create the Sequential model.
    Attributes:
        tr_x:
        n_a: Number of nodes into the hidden layer of each lstm cell.
    """
    np.random.seed(Config.SEED[0])
    tf.random.set_seed(Config.SEED[0])

    # "return_sequences=True" -> Return hidden state (activation function) output in each time step .
    # "return_state=True" -> Return the cell state
    # lstm1, state_h, state_c = LSTM(1, return_sequences=True, return_state=True)
    new_model = Sequential([
        Input(shape=tr_x.shape[-2:]),
        LSTM(n_a, return_sequences=True),
        Dropout(0.2),
        BatchNormalization(),
        LSTM(n_a),
        Dropout(0.2),
        BatchNormalization(),
        Dense(32, activation='sigmoid'),
        Dropout(0.2),
        Dense(1),
    ])  # 'softmax'

    # Adam optimization is a stochastic gradient descent method that is
    # based on adaptive estimation of first-order and second-order moments.
    opt = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    new_model.compile(optimizer=opt,
                      loss='mean_squared_error',
                      metrics=['mae']
                      )
    print(new_model.summary())

    return new_model


def create_callbacks(path, log_name, early_stop_patience=10):
    """"""
    tensorboard = TensorBoard(log_dir=path + log_name)

    filename = f"/{int(time.time())}" + "/RNN_Final-{epoch:02d}-{val_loss:.3f}"  # unique file name with epoch and validation acc for that epoch.
    checkpoint = ModelCheckpoint("{}{}.model".format(path, filename,
                                                     monitor='val_loss',
                                                     verbose=None,
                                                     save_best_only=True,
                                                     mode='max'))  # saves only the best ones.

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stop_patience,
                                                  restore_best_weights=True)

    ret_arr = [checkpoint, early_stop]  # ,tensorboard
    return ret_arr


def load_model(path):
    model = tf.keras.models.load_model(path)
    print(model.summary())
    return model


if __name__ == "__main__":
    config = Config()
    LOG_PATH = '/home/titoare/Documents/ds/g_research_cr/model/'
    PATH = "/home/titoare/Documents/ds/g_research_cr/input/j_data/btc.csv"

    """Reformat data to fit model"""
    import pandas as pd

    crypto_df = pd.read_csv("/home/titoare/Documents/ds/g_research_cr/input/j_data/btc_usd.csv", index_col=0)
    crypto_df = sequence_transform.DTSequence(crypto_df).process(15, 60)
    crypto_df.dropna(axis=0, inplace=True)
    train_x, train_y, test_x, test_y = crypto_processor.ProcessorSequence().process(crypto_df,
                                                                                    Config.TARGET_NAME,
                                                                                    Config.SEQ_LEN)
    """ Model creation """
    """
    - The batch size is a number of samples processed before the model is updated. When the loop goes through all
      the rows within each packet(batch), the model parameters are updated.
    - The number of epochs is the number of complete passes through the training dataset. Set how many times a loop goes
      through each batch.
    """
    if Config.TRAIN_MODEL:
        model = create_model(train_x, config.N_A)
        callbacks = create_callbacks(LOG_PATH, Config.NAME)
        history = model.fit(
            train_x, train_y,
            batch_size=Config.BATCH_SIZE,
            epochs=Config.EPOCHS,
            validation_data=(test_x, test_y),
            shuffle=False,
            callbacks=callbacks
        )
    else:
        path = "/home/titoare/Documents/ds/g_research_cr/model/1642695572/RNN_Final-17-0.000.model"
        model = load_model(path)
        del path



