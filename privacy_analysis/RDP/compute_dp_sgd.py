import math

from matplotlib import pyplot as plt

# from absl import app
#整个DP-SGD的隐私分析，要从这个开始函数调用。整体来看，隐私分析和数据不需要绑定，输入对应参数即可获得，
#从这个角度看，我们不需要每次迭代都调用这个隐私函数分析

#这个DPSGD是以epochs为参数做的
#这个函数调用下面的函数，这个函数主要判断q的
from privacy_analysis.RDP.compute_rdp import compute_rdp, _compute_rdp_randomized_response, \
    compute_rdp_randomized_response
from privacy_analysis.RDP.rdp_convert_dp import compute_eps, compute_eps2


def compute_dp_sgd_privacy(n, batch_size, noise_multiplier, epochs, delta):
    """Compute epsilon based on the given hyperparameters.
    Args:
      n: Number of examples in the training data.
      batch_size: Batch size used in training.
      noise_multiplier: Noise multiplier used in training.
      epochs: Number of epochs in training.
      delta: Value of delta for which to compute epsilon.
      S:sensitivity
    Returns:
      Value of epsilon corresponding to input hyperparameters.
    """
    q = batch_size / n
    if q > 1:
        print ('n must be larger than the batch size.')
    orders = (list(range(2, 64)) + [128, 256, 512])  # 默认的lamda

    steps = int(math.ceil(epochs * (n / batch_size)))

    return apply_dp_sgd_analysis(q, noise_multiplier, steps, orders, delta)

def apply_dp_sgd_analysis(q, sigma, steps, orders, delta):
  """Compute and print results of DP-SGD analysis."""


  rdp = compute_rdp(q, sigma, steps, orders)

  eps, opt_order = compute_eps(orders, rdp, delta)


  if opt_order == max(orders) or opt_order == min(orders):
    print('The privacy estimate is likely to be improved by expanding '
          'the set of orders.')

  return eps, opt_order

def RR_dp_privacy(p, steps, delta):
    orders = (list(range(2, 64)) + [128, 256, 512])

    rdp=compute_rdp_randomized_response(p,steps,orders)

    eps, opt_order = compute_eps(orders, rdp, delta)
    return eps,opt_order



if __name__=="__main__":
    sigma=0.5
    steps=1
    q=1.0
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64))+ [128, 256, 512]

    eps, opt_order = apply_dp_sgd_analysis(q,sigma, steps, orders, 10 ** (-5))
    print("eps:", format(eps) + "| order:", format(opt_order))
