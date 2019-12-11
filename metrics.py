from keras import backend as K
import tensorflow as tf
import numpy as np


def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    trs = np.linspace(0, 0.5)
    recallm = K.cast(0, y_pred.dtype)
    for threshold in trs:
        # P(falsealarm) + 19⋅P(miss), что равно FP/(FP+TN) + 19⋅FN/(FN+TP).
        threshold = K.cast(threshold, y_pred.dtype)
        y_pred_t = K.cast(y_pred > threshold, y_pred.dtype)
        true_positives = K.sum(K.round(K.clip(y_true * y_pred_t, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        recallm = K.maximum(recall, recallm)
    return recallm


def minC(y_true, y_pred):
    trs = np.linspace(0, 0.5)
    minCm = K.cast(1000.0, y_pred.dtype)
    for threshold in trs:
        # P(falsealarm) + 19⋅P(miss), что равно FP/(FP+TN) + 19⋅FN/(FN+TP).
        threshold = K.cast(threshold, y_pred.dtype)
        y_pred_t = K.cast(y_pred > threshold, y_pred.dtype)
        FP = K.sum(K.round(K.clip((1 - y_true) * y_pred_t, 0, 1)))
        TP = K.sum(K.round(K.clip(y_true * y_pred_t, 0, 1)))
        FN = K.sum(K.round(K.clip(y_true * (1 - y_pred_t), 0, 1)))
        TN = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred_t), 0, 1)))
        minC = FP / (FP + TN + K.epsilon()) + 19 * FN / (FN + TP + K.epsilon())
        minCm = K.minimum(minC, minCm)

    return minCm


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
