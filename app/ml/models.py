from __future__ import annotations

import tensorflow as tf


def build_multiclass_model() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(1, 7), name="kpi_sequence"),
            tf.keras.layers.LSTM(6, name="cause_lstm"),
            tf.keras.layers.Dense(7, activation="softmax", name="cause_probabilities"),
        ],
        name="fault_cause_classifier",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def build_binary_model() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(1, 7), name="kpi_sequence"),
            tf.keras.layers.LSTM(7, return_sequences=True, name="fault_lstm_sequence"),
            tf.keras.layers.Dropout(0.2, name="fault_dropout_sequence"),
            tf.keras.layers.LSTM(7, name="fault_lstm"),
            tf.keras.layers.Dropout(0.2, name="fault_dropout"),
            tf.keras.layers.Dense(1, activation="sigmoid", name="fault_probability"),
        ],
        name="binary_fault_detector",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model
