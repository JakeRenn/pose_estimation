#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import tensorflow as tf

ratio = 1.0
class_weights = np.array([
  1.0, 1.0, 1.0, 1.0, 1.0,
  1.0, 1.0, 1.0, 1.0, 1.0,
  1.0, 1.0, 1.0, 1.0, 1.0,
  1.0
  # 1.0, 10.0, 10.0, 10.0, 10.0,
  # 10.0, 10.0, 10.0, 10.0, 10.0,
  # 10.0, 10.0, 10.0, 10.0, 10.0,
  # 10.0
  ]).astype(np.float32).reshape([16])

def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
  # Need to generate a unique name to avoid duplicates:
  rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

  tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
  g = tf.get_default_graph()
  with g.gradient_override_map({"PyFunc": rnd_name}):
    return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

def _softmax_cross_entropy(predict, labels):
    scratch = np.max(predict, axis = -1)
    backprop = predict - np.expand_dims(scratch, axis = -1)
    scratch = np.sum(np.exp(backprop), axis=-1)
    loss = labels * (np.expand_dims(np.log(scratch), axis=-1) - backprop)
    loss = np.sum(loss, axis = -1)

    backprop = np.exp(backprop) / np.expand_dims(scratch, axis=-1) - labels

    return loss, backprop

def self_loss(predicts, labels):

    # shape = predicts.shape
    # predicts = predicts.reshape([-1, 20])
    # labels = labels.reshape([-1, 20])

    batch_label = np.argmax(labels, axis = -1).astype(np.float32)
    batch_zeros = np.zeros_like(batch_label).astype(np.float32)
    mask1 = np.not_equal(batch_zeros, batch_label)
    rand_u = np.random.uniform(low=0.0, high=1.0, size=batch_label.shape)
    mask2 = rand_u < ratio
    mask = mask1 | mask2

    loss, backprop = _softmax_cross_entropy(predicts, labels)
    loss = np.where(mask, loss, batch_zeros)
    backprop_zeros = np.zeros_like(backprop)
    backprop = np.where(np.expand_dims(mask, axis=-1), backprop, backprop_zeros)
    backprop = backprop * class_weights
    # backprop = np.reshape(backprop, shape)

    # loss = np.mean(loss)
    # return loss
    return loss, backprop

def custom_loss(predicts, labels, name=None):
  # with tf.op_scope([predicts, labels], name, "CustomLoss") as name:
    # loss, grad =  tf.py_func(self_loss, [predicts, labels],
        # [tf.float64, tf.float64], stateful=False, name="My")
  with tf.name_scope(name, "CustomLoss", [predicts, labels]) as name:

    loss, backprop = py_func(self_loss, [predicts, labels],
        [tf.float32, tf.float32], name=name,
        grad=_CustomLossGrad)
  # return tf.reduce_mean(loss)
  return loss


def _BroadcastMul(vec, mat):
  vec = tf.expand_dims(vec, -1)
  return vec*mat

# def _CustomLossGrad(op, grad_loss):
def _CustomLossGrad(op, grad_loss, grad_grad):
  softmax_grad = op.outputs[1]
  grad = _BroadcastMul(grad_loss, softmax_grad)

  if grad_grad.op.type not in ("ZerosLike", "Zeros"):
    logits = op.inputs[0]
    softmax = tf.nn.softmax(logits)
    grad += ((grad_grad - tf.squeeze(tf.matmul(grad_grad[:, None, :],
      softmax[:, :, None]), axis=1) * softmax))

  grad /= tf.cast(tf.size(grad) / tf.shape(grad)[-1], tf.float32)

  return grad, None

def main():
  with tf.Session() as sess:
    predicts = np.random.uniform(0.0, 1.0, (100, 20)).reshape((2, 10, 5,
      20)).astype(np.float32)
    # labels = np.zeros([200]).reshape((10, 20))
    tmp = np.random.randint(0, 20, size=(2, 10, 5)).astype(np.int32)
    # tmp[3] = 0
    # tmp[1] = 0
    labels = np.eye(20)[tmp].astype(np.float32)
    print labels.shape

    # lloss, ggrad = self_loss(predicts, labels)
    # loss = self_loss(predicts, labels)
    # print loss
    # print grad

    predicts = tf.constant(predicts)
    labels = tf.constant(labels)

    lloss=  custom_loss(predicts, labels)
    loss=  tf.losses.softmax_cross_entropy(labels, predicts)

    eval_lloss = lloss.eval()
    eval_loss = loss.eval()
    # print sess.run(loss)

    # print (tf.gradients(loss, predicts))
    print "Grad"
    # print ggrad
    my = tf.gradients(lloss, predicts)[0].eval()
    original = tf.gradients(loss, predicts)[0].eval()
    # my /= 10
    mask = np.isclose(my, original)
    print mask
    print mask.shape
    print eval_lloss
    print eval_loss
    # print a
    # print "----"
    # print b
    # print (tf.gradients(loss, predicts)[0].eval())



  # print (x.eval(), y.eval(), tf.gradients(y, x).eval())
    # print (predicts.eval(), labels.eval())

if __name__ == "__main__":
  main()
