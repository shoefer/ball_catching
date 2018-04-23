import numpy as np


class Strategy(object):
  """ Base class for strategies """

  control_type = "acceleration"
  # other possibilities: position -> set position directly

  def __init__(self, dicts):
    pass

  def start(self, **kwargs):
    pass

  def stop(self, **kwargs):
    pass

  def write_logs(self, log_root, trial):
    pass

  def step(self, i, x, dicts):
    raise NotImplementedError()

  @staticmethod
  def get_weights(strategy, dicts, W_name, W_shape, W=None):
    separator = "/"
    if strategy is None or strategy == "":
      separator = ""

    if len(W_shape) == 0:
      return dicts[strategy+separator+W_name]

    if W is None:
      W = np.zeros(W_shape)
    else:
      W[:] = 0

    for i in range(W_shape[0]):
      if len(W_shape) == 1:
        dict_ptn='%s%s%s_%d' % (strategy, separator, W_name, i)
        W[i] = np.array(dicts[dict_ptn])
      else:
        for j in range(W_shape[1]):
          dict_ptn='%s%s%s_%d,%d' % (strategy, separator, W_name, i, j)
          try:
            W[i,j] = np.array(dicts[dict_ptn])
          except:
            warn("%s could not be parsed" % dict_ptn)

    return W