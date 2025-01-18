"""arcface.py
"""
import tensorflow as tf

class ArcFace():
    def __init__(self, s=30.0, m=0.5):
        self.s = s
        self.m = m

    def __call__(self, y_true, y_pred):
        
        # y_pred = tf.nn.l2_normalize(y_pred, axis=1)
        # y_true = tf.nn.l2_normalize(y_true, axis=1)
        # cos_m = tf.math.cos(self.m)
        # sin_m = tf.math.sin(self.m)
        # th = tf.math.cos(tf.constant(math.pi) - self.m)
        # mm = tf.math.sin(tf.constant(math.pi) - self.m) * self.m
        # threshold = tf.math.cos(tf.constant(math.pi) - self.m)
        # cond_v = y_pred - threshold
        # cond = tf.cast(tf.nn.relu(cond_v), dtype=tf.bool)
        # keep_val = self.s * (y_pred - mm)
        # cos_t = y_pred
        # new_zy = tf.where(cond, keep_val, cos_t)
        # final_zy = new_zy * y_true
        # return tf.nn.softmax(final_zy * self.s)
