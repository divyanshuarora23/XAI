"""
model.py
--------
CNN-BiLSTM architecture for IoT network traffic classification.

Input  : (batch, WINDOW_SIZE, n_features)  — 3-D tensor
Output : (batch, 1)                         — sigmoid probability
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_cnn_bilstm(input_shape: tuple, dropout_rate: float = 0.3) -> tf.keras.Model:
    """
    Build the CNN-BiLSTM model.

    Args:
        input_shape  : (window_size, n_features)
        dropout_rate : dropout applied after each major block

    Returns:
        Compiled Keras model
    """
    inputs = tf.keras.Input(shape=input_shape, name='traffic_input')

    # ── CNN Block ─────────────────────────────────────────────────────────────
    # Extracts local spatial patterns across features within each time step
    x = layers.Conv1D(
        filters=64, kernel_size=3, padding='same',
        activation='relu',
        kernel_regularizer=regularizers.l2(1e-4),
        name='conv1'
    )(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.MaxPooling1D(pool_size=2, name='pool1')(x)
    x = layers.Dropout(dropout_rate, name='drop_cnn1')(x)

    x = layers.Conv1D(
        filters=128, kernel_size=3, padding='same',
        activation='relu',
        kernel_regularizer=regularizers.l2(1e-4),
        name='conv2'
    )(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.MaxPooling1D(pool_size=2, name='pool2')(x)
    x = layers.Dropout(dropout_rate, name='drop_cnn2')(x)

    # ── BiLSTM Block ──────────────────────────────────────────────────────────
    # Captures temporal dependencies in both directions
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=dropout_rate,
                    recurrent_dropout=0.1),
        name='bilstm1'
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=False, dropout=dropout_rate,
                    recurrent_dropout=0.1),
        name='bilstm2'
    )(x)

    # ── Classifier Head ───────────────────────────────────────────────────────
    x = layers.Dense(128, activation='relu',
                    kernel_regularizer=regularizers.l2(1e-4),
                    name='dense1')(x)
    x = layers.Dropout(dropout_rate, name='drop_dense')(x)
    x = layers.Dense(64, activation='relu', name='dense2')(x)
    output = layers.Dense(1, activation='sigmoid', name='output')(x)

    model = models.Model(inputs=inputs, outputs=output, name='CNN_BiLSTM_IoT')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
        ]
    )

    return model


def get_callbacks(checkpoint_path: str = 'model/artifacts/best_model.keras'):
    """
    Standard callbacks for training.
    - EarlyStopping   : stops if val_loss doesn't improve for 5 epochs
    - ModelCheckpoint : saves best weights automatically
    - ReduceLROnPlateau: halves LR if val_loss stalls for 3 epochs
    """
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss', save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=3, min_lr=1e-6, verbose=1
        ),
    ]