from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import optimizer


class AMSGrad(optimizer.Optimizer):
  """The AMSGrad algorithm in the paper, On the Convergence of Adam and Beyond"""

  def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999,
               epsilon=1e-8, use_locking=False, name="AMSGrad"):
    super(AMSGrad, self).__init__(use_locking, name)
    self._lr = learning_rate
    self._beta1 = beta1
    self._beta2 = beta2
    self._epsilon = epsilon

    self._lr_t = None
    self._beta1_t = None
    self._beta2_t = None
    self._epsilon_t = None

    self._beta1_power = None
    self._beta2_power = None

  def _create_slots(self, var_list):

    first_var = min(var_list, key=lambda x: x.name)

    create_new = self._beta1_power is None
    if not create_new and context.in_graph_mode():
      create_new = (self._beta1_power.graph is not first_var.graph)

    if create_new:
      with ops.colocate_with(first_var):
        self._beta1_power = variable_scope.variable(self._beta1,
                                                    name="beta1_power",
                                                    trainable=False)
        self._beta2_power = variable_scope.variable(self._beta2,
                                                    name="beta2_power",
                                                    trainable=False)
    # Create slots for the first and second moments.
    for v in var_list:
      # first moment est
      self._zeros_slot(v, "first_mom", self._name)
      # second moment est
      self._zeros_slot(v, "second_mom", self._name)
      self._zeros_slot(v, "second_mom_max", self._name)

  def _prepare(self):
    self._lr_t = ops.convert_to_tensor(self._lr)
    self._beta1_t = ops.convert_to_tensor(self._beta1)
    self._beta2_t = ops.convert_to_tensor(self._beta2)
    self._epsilon_t = ops.convert_to_tensor(self._epsilon)
    self._one_minus_beta1 = ops.convert_to_tensor(1. - self._beta1)
    self._one_minus_beta2 = ops.convert_to_tensor(1. - self._beta2)

  def _apply_dense(self, grad, var):
    # bias-corrected learning rate
    lr = self._lr_t * math_ops.sqrt(1. - self._beta2_power) / (1. - self._beta1_power)
    first_mom = self.get_slot(var, "first_mom")
    second_mom = self.get_slot(var, "second_mom")
    second_mom_max = self.get_slot(var, "second_mom_max")
    first_update = first_mom.assign(self._beta1_t * first_mom +
                                    self._one_minus_beta1 * grad,
                                    use_locking=self._use_locking)
    second_update = second_mom.assign(self._beta2_t * second_mom +
                                      self._one_minus_beta2 * math_ops.square(grad),
                                      use_locking=self._use_locking)
    # AMSGrad compared to ADAM
    second_max_update = second_mom_max.assign(gen_math_ops.maximum(second_mom_max,
                                                                   second_update))
    var_update = var.assign_sub(lr * first_update / (math_ops.sqrt(second_max_update) +
                                                     self._epsilon_t),
                                use_locking=self._use_locking)
    return control_flow_ops.group(*[var_update, first_update,
                                    second_update, second_max_update])

  def _apply_sparse(self, grad, var):
    # just a copy of the dense case, not properly implemented yet
    return self._apply_dense(grad, var)

  def _finish(self, update_ops, name_scope):
    # Update the power accumulators.
    with ops.control_dependencies(update_ops):
      with ops.colocate_with(self._beta1_power):
        update_beta1 = self._beta1_power.assign(
          self._beta1_power * self._beta1,
          use_locking=self._use_locking)
        update_beta2 = self._beta2_power.assign(
          self._beta2_power * self._beta2_t,
          use_locking=self._use_locking)
    return control_flow_ops.group(*update_ops + [update_beta1, update_beta2],
                                  name=name_scope)


class Adam(optimizer.Optimizer):
  """Adam"""

  def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999,
               epsilon=1e-8, use_locking=False, name="Adam"):
    super(Adam, self).__init__(use_locking, name)
    self._lr = learning_rate
    self._beta1 = beta1
    self._beta2 = beta2
    self._epsilon = epsilon

    self._lr_t = None
    self._beta1_t = None
    self._beta2_t = None
    self._epsilon_t = None

    self._beta1_power = None
    self._beta2_power = None

  def _create_slots(self, var_list):

    first_var = min(var_list, key=lambda x: x.name)

    create_new = self._beta1_power is None
    if not create_new and context.in_graph_mode():
      create_new = (self._beta1_power.graph is not first_var.graph)

    if create_new:
      with ops.colocate_with(first_var):
        self._beta1_power = variable_scope.variable(self._beta1,
                                                    name="beta1_power",
                                                    trainable=False)
        self._beta2_power = variable_scope.variable(self._beta2,
                                                    name="beta2_power",
                                                    trainable=False)
    # Create slots for the first and second moments.
    for v in var_list:
      # first moment est
      self._zeros_slot(v, "first_mom", self._name)
      # second moment est
      self._zeros_slot(v, "second_mom", self._name)

  def _prepare(self):
    self._lr_t = ops.convert_to_tensor(self._lr)
    self._beta1_t = ops.convert_to_tensor(self._beta1)
    self._beta2_t = ops.convert_to_tensor(self._beta2)
    self._epsilon_t = ops.convert_to_tensor(self._epsilon)
    self._one_minus_beta1 = ops.convert_to_tensor(1. - self._beta1)
    self._one_minus_beta2 = ops.convert_to_tensor(1. - self._beta2)

  def _apply_dense(self, grad, var):
    # bias-corrected learning rate
    lr = self._lr_t * math_ops.sqrt(1. - self._beta2_power) / (1. - self._beta1_power)
    first_mom = self.get_slot(var, "first_mom")
    second_mom = self.get_slot(var, "second_mom")
    first_update = first_mom.assign(self._beta1_t * first_mom +
                                    self._one_minus_beta1 * grad,
                                    use_locking=self._use_locking)
    second_update = second_mom.assign(self._beta2_t * second_mom +
                                      self._one_minus_beta2 * math_ops.square(grad),
                                      use_locking=self._use_locking)
    var_update = var.assign_sub(lr * first_update / (math_ops.sqrt(second_update) +
                                                     self._epsilon_t),
                                use_locking=self._use_locking)
    return control_flow_ops.group(*[var_update, first_update, second_update])

  def _apply_sparse(self, grad, var):
    # just a copy of the dense case, not properly implemented yet
    return self._apply_dense(grad, var)

  def _finish(self, update_ops, name_scope):
    # Update the power accumulators.
    with ops.control_dependencies(update_ops):
      with ops.colocate_with(self._beta1_power):
        update_beta1 = self._beta1_power.assign(
          self._beta1_power * self._beta1,
          use_locking=self._use_locking)
        update_beta2 = self._beta2_power.assign(
          self._beta2_power * self._beta2_t,
          use_locking=self._use_locking)
    return control_flow_ops.group(*update_ops + [update_beta1, update_beta2],
                                  name=name_scope)

