# -*- coding: utf-8 -*-
"""bci_model.py -- 언니들 모델(Conv1D+LSTM+MHSA) 정의. train_mixed_5fold.py 와 동일, 복사만 했다."""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
import tensorflow.keras.layers as _layers

# ===== train_mixed.py 와 같은 안정화 패치 =====
_orig_lstm_init = _layers.LSTM.__init__
def _patched_lstm_init(self, *args, **kwargs):
    kwargs.pop("time_major", None)
    _orig_lstm_init(self, *args, **kwargs)
_layers.LSTM.__init__ = _patched_lstm_init


# ===== 커스텀 레이어 / 모델 (train_mixed.py 와 동일) =====
@tf.keras.utils.register_keras_serializable()
class PositionalEncodingLayer(layers.Layer):
    def __init__(self, max_pos, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.max_pos = max_pos; self.embed_dim = embed_dim
        pos = np.arange(max_pos)[:, None]
        div = np.exp(np.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim))
        pe = np.zeros((max_pos, embed_dim))
        pe[:, 0::2] = np.sin(pos * div); pe[:, 1::2] = np.cos(pos * div)
        self.pos_encoding = tf.constant(pe, dtype=tf.float32)
    def call(self, x):
        return x + self.pos_encoding[None, :tf.shape(x)[1], :]
    def get_config(self):
        return {"max_pos": self.max_pos, "embed_dim": self.embed_dim, **super().get_config()}


@tf.keras.utils.register_keras_serializable()
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim; self.num_heads = num_heads
        self.proj_dim = embed_dim // num_heads
        self.q = layers.Dense(embed_dim); self.k = layers.Dense(embed_dim)
        self.v = layers.Dense(embed_dim); self.out = layers.Dense(embed_dim)
    def split_heads(self, x, b):
        return tf.transpose(tf.reshape(x, (b, -1, self.num_heads, self.proj_dim)), [0, 2, 1, 3])
    def call(self, x):
        b = tf.shape(x)[0]
        q = self.split_heads(self.q(x), b); k = self.split_heads(self.k(x), b); v = self.split_heads(self.v(x), b)
        score = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.proj_dim, tf.float32))
        w = tf.nn.softmax(score, axis=-1)
        att = tf.matmul(w, v); att = tf.transpose(att, [0, 2, 1, 3])
        concat = tf.reshape(att, (b, -1, self.embed_dim))
        return self.out(concat)
    def get_config(self):
        return {"embed_dim": self.embed_dim, "num_heads": self.num_heads, **super().get_config()}


def LN():
    return tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)


def build_hybrid_model(window_size=256, channels=4, bands=4, heads=4, l2=3e-4):
    inp = layers.Input(shape=(window_size, channels, bands))
    x = layers.Lambda(lambda t: tf.reshape(t, (tf.shape(t)[0], tf.shape(t)[1],
                                               tf.shape(t)[2] * tf.shape(t)[3])))(inp)
    x = layers.Conv1D(64, 1, padding="same", activation="relu", kernel_regularizer=regularizers.l2(l2))(x)
    x = LN()(x); x = layers.Dropout(0.3)(x)
    x = layers.Conv1D(128, 1, padding="same", activation="relu", kernel_regularizer=regularizers.l2(l2))(x)
    x = LN()(x); x = layers.Dropout(0.3)(x)
    x = layers.Conv1D(64, 1, padding="same", activation="relu", kernel_regularizer=regularizers.l2(l2))(x)
    x = _layers.LSTM(64, return_sequences=True, dropout=0.3)(x)
    x = PositionalEncodingLayer(max_pos=window_size, embed_dim=64)(x)
    x = MultiHeadSelfAttention(embed_dim=64, num_heads=heads)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inp, out, name="hybrid_ln_lstm_mhsa")


# ===== 전처리 (train_mixed.py 와 동일) =====
FS = 256


def augment_numpy(X, y, times=1, noise_std=0.02, ch_drop_p=0.1, max_shift_frac=0.05, drift_max_abs=0.2, band_drop_p=0.15, seed=42):
    """원본 + 증강본 (train_mixed_5fold.py 와 동일)"""
    if times <= 0 or X.shape[0] == 0:
        return X, y
    rng = np.random.RandomState(seed)
    N, T, C, B = X.shape
    outs = [X]; ys = [y]
    for _ in range(times):
        x2 = X.copy()
        band_std = np.std(x2.reshape(-1, B), axis=0) + 1e-6
        x2 = x2 + (rng.randn(*x2.shape) * (noise_std * band_std[None, None, None, :])).astype(np.float32)
        mask = np.ones((N, C), dtype=np.float32)
        for i in range(N):
            if rng.rand() < ch_drop_p:
                mask[i, rng.randint(0, C)] = 0.0
        x2 = x2 * mask[:, None, :, None]
        ms = max(1, int(round(T * max_shift_frac)))
        for i, s in enumerate(rng.randint(-ms, ms + 1, size=(N,))):
            x2[i] = np.roll(x2[i], shift=s, axis=0)
            end = rng.uniform(-drift_max_abs, drift_max_abs)
            x2[i] = x2[i] + np.linspace(0.0, end, T, dtype=np.float32)[:, None, None]
            x2[i] = x2[i] * (rng.rand(1, 1, B) > band_drop_p).astype(np.float32)
        outs.append(x2); ys.append(y)
    return np.vstack(outs).astype(np.float32), np.hstack(ys).astype(np.int32)