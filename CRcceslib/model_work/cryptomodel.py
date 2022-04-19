from CRcceslib.data_processor import crypto_processor
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
    EPOCHS = 50
    BATCH_SIZE = 64
    TARGET_NAME = "return_15min"
    NAME = f"{SEQ_LEN}_SEQ_{FUTURE_PERIOD_PREDICT}_PRED_{int(time.time())}"


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
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='softmax'),
    ])

    # Adam optimization is a stochastic gradient descent method that is
    # based on adaptive estimation of first-order and second-order moments.
    opt = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
    new_model.compile(optimizer=opt,
                      loss='mae',
                      metrics=['accuracy']
                      )
    print(new_model.summary())

    return new_model


def create_callbacks(path, log_name):
    """"""
    tensorboard = TensorBoard(log_dir=path + log_name)

    filename = "RNN_Final-{epoch:02d}-{val_accuracy:.3f}"  # unique file name with epoch and validation acc for that epoch.
    checkpoint = ModelCheckpoint("{}{}.model".format(path, filename,
                                                     monitor='val_accuracy',
                                                     verbose=1,
                                                     save_best_only=True,
                                                     mode='max'))  # saves only the best ones.

    return [tensorboard, checkpoint]


if __name__ == "__main__":
    config = Config()
    LOG_PATH = '/home/titoare/Documents/ds/g_research_cr/model/'
    PATH = "/home/titoare/Documents/ds/g_research_cr/input/j_data/btc.csv"

    """Reformat data to fit model"""
    import pandas as pd
    crypto_df = pd.read_csv(PATH, index_col=0)
    crypto_df.dropna(axis=0, inplace=True)
    crypto_df.drop("time", inplace=True, axis=1)  # Only with the doc saved, in teh real case time is deleted.
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
    model = create_model(train_x, config.N_A)
    callbacks = create_callbacks(LOG_PATH, Config.NAME)
    history = model.fit(
        train_x, train_y,
        batch_size=Config.BATCH_SIZE,
        epochs=Config.EPOCHS,
        validation_data=(test_x, test_y),

    )
    callbacks=callbacks

