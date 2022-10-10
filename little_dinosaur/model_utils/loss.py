import numpy as np
from keras import backend as K
# import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

def cross_entropy(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)


def symmetric_cross_entropy(alpha, beta):
    """
    https://github.com/YisenWang/symmetric_cross_entropy_for_noisy_labels

    How to regulate the alpha and the beta of SCEloss?
    See https://github.com/YisenWang/symmetric_cross_entropy_for_noisy_labels/issues/5

    If the labels/mask are clean, try 1*CE + beta*RCE, beta can be selected to make sure the two loss terms have similar magnitudes. 
    In terms of segmentation, it is related to the sparsity of the mask. Sometimes, the value of beta*RCE can be 10x large than the CE term, which may produce better performance.
    If the labels/mask are noisy, try around 0.1*CE + beta*RCE. The 0.1*CE is to reduce the overfitting of CE to noisy mask. The beta can be large such as 6/10 to boost training.
    If the dataset is very sparse, and has a convergence problem. The try large alpha [1, 10]. For example: 6*CE+ beta*RCE. The beta can be determined similarly as above.


    """

    def loss(y_true, y_pred):
        y_true_1 = y_true
        y_pred_1 = y_pred

        y_true_2 = y_true
        y_pred_2 = y_pred

        y_pred_1 = tf.clip_by_value(y_pred_1, 1e-7, 1.0)
        y_true_2 = tf.clip_by_value(y_true_2, 1e-4, 1.0)

        return alpha * tf.reduce_mean(-tf.reduce_sum(
            y_true_1 * tf.log(y_pred_1), axis=-1)) + beta * tf.reduce_mean(
                -tf.reduce_sum(y_pred_2 * tf.log(y_true_2), axis=-1))

    return loss


def lsr(K):
    """
    Rethinking the Inception Architecture for Computer Vision
    https://arxiv.org/pdf/1512.00567.pdf

    We propose a mechanism for encouraging the model to
    be less confident. While this may not be desired if the goal
    is to maximize the log-likelihood of training labels, it does
    regularize the model and makes it more adaptable

    """

    def loss(y_true, y_pred):
        epsilon = 0.1
        y_smoothed_true = y_true * (1 - epsilon - epsilon / float(K))
        y_smoothed_true = y_smoothed_true + epsilon / float(K)

        y_pred_1 = tf.clip_by_value(y_pred, 1e-7, 1.0)

        return tf.reduce_mean(
            -tf.reduce_sum(y_smoothed_true * tf.log(y_pred_1), axis=-1))

    return loss


def generalized_cross_entropy(y_true, y_pred):
    """
    2018 - nips - Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels.
    Being a nonsymmetric and unbounded loss function, CCE is sensitive to label noise. On the contrary,
    MAE, as a symmetric loss function, is noise robust. 

    MSE,MAE -- symmetric

    when training with CCE,
    more emphasis is put on difficult samples. This implicit weighting scheme is desirable for training
    with clean data, but can cause overfitting to noisy labels

    MAE treats every sample equally, which makes it more robust to noisy labels.
    However, as we demonstrate empirically, this can lead to significantly longer training time before
    convergence. Moreover, without the implicit weighting scheme to focus on challenging samples, the
    stochasticity involved in the training process can make learning difficult. As a result, classification
    accuracy might suffer

    To exploit the benefits of both the noise-robustness provided by MAE and the implicit weighting
    scheme of CCE, we propose using the the negative Box-Cox transformation as a loss function

    Lq = (1 - f(x)**q)/q

    To learn more Box-Cox transformation

    George EP Box and David R Cox. An analysis of transformations. Journal of the Royal
    Statistical Society. Series B (Methodological), pages 211–252, 1964


    """
    q = 0.4
    t_loss = (1 - tf.pow(tf.reduce_sum(y_true * y_pred, axis=-1), q)) / q
    return tf.reduce_mean(t_loss)


def joint_optimization_loss(K):
    """
    2018 - cvpr - Joint optimization framework for learning with noisy labels.

    we experimentally found that a high learning rate suppresses the memorization 
    ability of a DNN and prevents it from completely fitting to labels.
    Thus, we assume that a network trained with a high learning rate will have more
    difficulty fitting to noisy labels. In other words, the loss is high for noisy 
    labels and low for clean labels

    similar paper: Probabilistic End-to-end Noise Correction for Learning with Noisy Labels

    """

    def loss(y_true, y_pred):
        y_pred_avg = K.mean(y_pred, axis=0)
        p = np.ones(15, dtype=np.float32) / 15.
        l_p = -K.sum(K.log(y_pred_avg) * p)
        l_e = K.categorical_crossentropy(y_pred, y_pred)
        return K.categorical_crossentropy(y_true,
                                          y_pred) + 1.2 * l_p + 0.8 * l_e

    return loss


def boot_soft(y_true, y_pred):
    """
    2015 - iclrws - Training deep neural networks on noisy labels with bootstrapping.

    By paying less heed to inconsistent labels, the learner can develop a more coherent model, which
    further improves its ability to evaluate the consistency of noisy labels. We refer to this approach
    as “bootstrapping”, in the sense of pulling oneself up by one’s own bootstraps, and also due to
    inspiration from the work of Yarowsky (1995) which is also referred to as bootstrapping.
    Concretely, we use a cross-entropy objective as before, but generate new regression targets for each
    SGD mini-batch based on the current state of the model.

    “Soft” bootstrapping uses predicted class probabilities q directly to generate regression 
    targets for each batch as follows
    
    """
    beta = 0.95

    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    return -K.sum(
        (beta * y_true + (1. - beta) * y_pred) * K.log(y_pred), axis=-1)


def boot_hard(y_true, y_pred):
    """
    2015 - iclrws - Training deep neural networks on noisy labels with bootstrapping.

    By paying less heed to inconsistent labels, the learner can develop a more coherent model, which
    further improves its ability to evaluate the consistency of noisy labels. We refer to this approach
    as “bootstrapping”, in the sense of pulling oneself up by one’s own bootstraps, and also due to
    inspiration from the work of Yarowsky (1995) which is also referred to as bootstrapping.
    Concretely, we use a cross-entropy objective as before, but generate new regression targets for each
    SGD mini-batch based on the current state of the model.

    “Hard” bootstrapping modifies regression targets using the MAP estimate of q given x, which we
    denote as zk := 1[k = argmax qi, i = 1...L]:
    """
    beta = 0.8

    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    pred_labels = K.one_hot(K.argmax(y_pred, 1),
                            num_classes=K.shape(y_true)[1])
    return -K.sum(
        (beta * y_true + (1. - beta) * pred_labels) * K.log(y_pred), axis=-1)


def forward(P):
    """
    Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach.
    """
    P = K.constant(P)

    def loss(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        return -K.sum(y_true * K.log(K.dot(y_pred, P)), axis=-1)

    return loss


def backward(P):
    """
    Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach.
    """
    P_inv = K.constant(np.linalg.inv(P))

    def loss(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        return -K.sum(K.dot(y_true, P_inv) * K.log(y_pred), axis=-1)

    return loss


def focal_loss(gamma=2., alpha=.25):
    """
    https://github.com/mkocabas/focal-loss-keras
    https://arxiv.org/abs/1708.02002

    FL(p_t)=- (1 - p_t) ^ gamma * log (p_t)

    We propose a novel loss we term the Focal Loss that
    adds a factor (1 − pt)γ to the standard cross entropy criterion.
    Setting γ > 0 reduces the relative loss for well-classified examples
    (pt > .5), putting more focus on hard, misclassified examples. As
    our experiments will demonstrate, the proposed focal loss enables
    training highly accurate dense object detectors in the presence of
    vast numbers of easy background examples

    Binary classification models are by default initialized to
    have equal probability of outputting either y = −1 or 1.
    Under such an initialization, in the presence of class imbalance, the loss due to the frequent class can dominate total
    loss and cause instability in early training. To counter this,
    we introduce the concept of a ‘prior’ for the value of p estimated by the model for the rare class (foreground) at the
    start of training. We denote the prior by π and set it so that
    the model’s estimated p for examples of the rare class is low,
    e.g. 0.01. We note that this is a change in model initialization (see §4.1) and not of the loss function. We found this
    to improve training stability for both the cross entropy and
    focal loss in the case of heavy class imbalance

    """

    def loss(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) *
                       K.log(pt_1 + K.epsilon())) - K.mean(
                           (1 - alpha) * K.pow(pt_0, gamma) *
                           K.log(1. - pt_0 + K.epsilon()))

    return loss
