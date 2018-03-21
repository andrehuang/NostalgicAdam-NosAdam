from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.util.tf_export import tf_export

def exponential_decay(learning_rate,
                      global_step,  # t, epoch
                      decay_steps,
                      decay_rate,
                      staircase=False,
                      name=None):
  """Applies exponential decay to the beta

 It requires a `global_step` value to
  compute the decayed learning rate.  You can just pass a TensorFlow variable
  that you increment at each training step.

  The function returns the decayed learning rate.  It is computed as:

  ```python
  decayed_learning_rate = learning_rate *
                          decay_rate ^ (global_step / decay_steps)
  ```

  If the argument `staircase` is `True`, then `global_step / decay_steps` is an
  integer division and the decayed learning rate follows a staircase function.

  Args:
    learning_rate: A scalar `float32` or `float64` `Tensor` or a
      Python number.  The initial learning rate.
    global_step: A scalar `int32` or `int64` `Tensor` or a Python number.
      Global step to use for the decay computation.  Must not be negative.
    decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
      Must be positive.  See the decay computation above.
    decay_rate: A scalar `float32` or `float64` `Tensor` or a
      Python number.  The decay rate.
    staircase: Boolean.  If `True` decay the learning rate at discrete intervals
    name: String.  Optional name of the operation.  Defaults to
      'ExponentialDecay'.

  Returns:
    A scalar `Tensor` of the same type as `learning_rate`.  The decayed
    learning rate.

  Raises:
    ValueError: if `global_step` is not supplied.
  """
  if global_step is None:
    raise ValueError("global_step is required for exponential_decay.")
  with ops.name_scope(
      name, "betaExponentialDecay",
      [learning_rate, global_step, decay_steps, decay_rate]) as name:
    learning_rate = ops.convert_to_tensor(learning_rate, name="learning_rate")
    dtype = learning_rate.dtype
    global_step = math_ops.cast(global_step, dtype)
    decay_steps = math_ops.cast(decay_steps, dtype)
    decay_rate = math_ops.cast(decay_rate, dtype)
    p = global_step / decay_steps
    if staircase:
      p = math_ops.floor(p)
    return math_ops.multiply(
        learning_rate, math_ops.pow(decay_rate, p), name=name)

def ratio_decay(alpha,
                global_step = 1,
                f_old = 0,
                f_new = 1,
                name=None):
    # if alpha < 0:
    #     raise ValueError("It's required that alpha >= 0")

    if global_step is None:
        raise ValueError("global_step is required for ratio_decay.")

    with ops.name_scope(
            name, "betaRatioDecay",
            [alpha, global_step, f_old, f_new]) as name:

        alpha = ops.convert_to_tensor(alpha)
        dtype = alpha.dtype
        global_step = math_ops.cast(global_step, dtype)
        f_old = math_ops.cast(f_old, dtype)
        f_new = math_ops.cast(f_new, dtype)

        beta2 = math_ops.divide(f_old, f_new)

        # f_old = f_old + math_ops.pow(global_step, -alpha)
        # f_new = f_new + math_ops.pow(global_step+1, -alpha)
        # f_old = f_old + 1
        # f_new = f_new + 1

        return beta2

    return


        # decay_steps = math_ops.cast(decay_steps, dtype)
        # decay_rate = math_ops.cast(decay_rate, dtype)
        # p = global_step / decay_steps

        # beta2 = control_flow_ops.cond(
        #     math_ops.equal(global_step, 1), lambda: 1.0,
        #     lambda: math_ops.ceil(global_step / decay_steps))


        # 不要用recursion定义了。只要存下上一步的值就可以。
        # def f(t):
        #     # print(tf.sign(t))
        #     # t is a tensor, can't be defined like this
        #     assert tf.minimum(t, 1) == 1
        #
        #     def g(t):
        #         return f(t - 1) + t ^ (-alpha)
        #     # print(tf.equal(t,1))
        #     a = tf.cond(tf.equal(t, 1), lambda: 1,  g(t))
        #
        #
        #     # if tf.equal(t, 1):
        #     #     return 1
        #     # else:
        #     #     return f(t - 1) + t ^ (-alpha)
        #     print(a)
        #     return a

        # global numerator, denominator
        #
        # if global_step == 0:
        #     numerator = 0
        #     denominator = 1
        # if global_step > 0:
        #     numerator = denominator

