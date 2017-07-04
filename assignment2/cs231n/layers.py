import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  D = w.shape[0] #no need to calculate D as as have that info from w
  #D = np.prod(x.shape[1:]) #this works too if we want to explicitly calculate it
  x_reshaped = np.reshape(x,(-1,D))
  out = np.dot(x_reshaped,w)+b
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)
    - b: Biases, of shape (M,)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################

  D = np.prod(x.shape[1:])
  x_reshaped = np.reshape(x,(-1,D))
  dodx = w
  dodw = x_reshaped
  dodb = 1
  dx = np.dot(dout,dodx.T)
  dx = np.reshape(dx,(-1,*x.shape[1:]))
  dw = np.dot(dodw.T,dout) #gradients add at branches and so we don't have to take the mean, just need to sum them for each example
  db = dout*dodb
  db = np.sum(db,axis=0) #gradients add at branches and so we don't have to take the mean, just need to sum them for each example
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  out = np.array(x, copy=True)
  out[out<0]=0
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  dx = np.ones_like(x)
  dx[x<=0]=0
  dx *= dout
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  
  if mode == 'train':
      
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
      sample_mean = 1/N*np.sum(x,axis=0) #Dx1
      xmu = x-sample_mean #NxD
      sample_var = 1/N*np.sum((xmu)**2,axis=0) #Dx1
      sqrtvar = np.sqrt(sample_var+eps) #Dx1
      x_norm = xmu/sqrtvar #NxD
      out = x_norm*gamma + beta
    
      running_mean = momentum*running_mean + (1-momentum)*sample_mean
      running_var = momentum*running_var + (1-momentum)*sample_var

      
      cache = (x,gamma,beta,xmu,sqrtvar)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    x_norm = (x-running_mean)/np.sqrt(running_var)
    out = x_norm*gamma + beta
    
    cache = None #this is redundant but added for better readability

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  #no need to return bn_param as it is mutable 
  #so it will automatically be modified outside this function in the calling function
  
  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################
  x,gamma,beta,xmu,sqrtvar = cache  
  N,D = x.shape

  dLdbeta = dout * (np.ones((1,D))) #dydbeta is broadcasted from 1xD to NxD
  dbeta = np.sum(dLdbeta,axis=0) #Dx1
  
  dLdgamma = dout * (xmu/sqrtvar) #NxD
  dgamma = np.sum(dLdgamma,axis=0) #Dx1

  dLdxh = dout*gamma #NxD
  dLdivar = dLdxh * xmu #NxD
  dLdivar = np.sum(dLdivar,axis=0) #Dx1
  dLdsqrtvar = dLdivar * -1*(sqrtvar**(-2)) #Dx1
  dLdvar = dLdsqrtvar * 1/2/sqrtvar #Dx1
  dLd04 = 1/N*dLdvar #Dx1 but need to broadcast it to NxD. Next shttps://www.bloomberg.com/view/articles/2017-02-24/what-google-hopes-to-gain-by-suing-ubertep takes care of it.
  dLd05_p1 = dLd04*2*xmu #NxD
  dLd05_p2 = dLdxh * 1/sqrtvar #NxD
  dLd05 = dLd05_p1 + dLd05_p2 #NxD
  dLdx_p1 = dLd05 #NxD
  dLd06 = np.sum(dLd05,axis=0) #Dx1
  dLd06 = -1*dLd06 #Dx1
  dLdx_p2 = 1/N*dLd06 #Dx1 but needs to be broadcasted to NxD. Next step takes care of it.
  dx = dLdx_p1 + dLdx_p2 #NxD
  

#  dydxN = gamma #Dx1
#  a = x-sample_mean #NxD
#  b = np.sqrt(sample_var+eps) #Dx1
#  #dadx = 1-1/N
#  #dbdx = 1/N*(1-1/N)*a/b #NxD. b will be broadcasted from Dx1 to NxD
#  dxNdx = (1-1/N)*(b-(a**2)/(b*N))/(b**2) #NxD
#  dydx = dydxN * dxNdx #NxD
#  dx = dout * dydx #N*D
  
#  dx = (1. / N) * gamma * (sample_var + eps)**(-1. / 2.) * (N * dout - np.sum(dout, axis=0) - 
#        (x - sample_mean) * (sample_var + eps)**(-1.0) * np.sum(dout * (x - sample_var), axis=0))

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    mask = (np.random.rand(*x.shape) < p) / p #dropout mask
    out = x * mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    out = x
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    dx = dout * mask #the derivates also scale by 1/p
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  s = conv_param['stride']
  p = conv_param['pad']
  
  x_pad = np.lib.pad(x,((0,0),(0,0),(p,p),(p,p)),'constant',constant_values=(0,0))

  #input shape
  N = x.shape[0]
  C = x.shape[1]
  H = x.shape[2]
  W = x.shape[3]

  #filter shape
  F = w.shape[0]
  #C is same as above
  HH = w.shape[2]
  WW = w.shape[3]
  w_reshaped = np.reshape(w,(-1,F))
  #no need to reshape b as it is of shape (F,)
  
  #output shape
  #N is same as above
  #F is same as above (F will be the number of channels)
  H_prime = int(1 + (H+2*p-HH)/s)
  W_prime = int(1 + (W+2*p-WW)/s)
  
  out_shape = (N,F,H_prime,W_prime)
  out = np.zeros(out_shape)

#NOT SURRE WHY THIS IS NOT WORKING  
#  for i in range(0,H_prime,1):
#      for j in range(0,W_prime,1):
#          h_strt = i*s 
#          h_stp = h_strt + HH
#          w_strt = j*s
#          w_stp = w_strt + WW
#          x_tmp = x_pad[:,:,h_strt:h_stp,w_strt:w_stp]
#          #print(x_tmp.shape)
#          #print(w_reshaped.shape)
#          out_tmp, _ = affine_forward(x_tmp,w_reshaped,b) #out_tmp will be of shape NxF
#          #out_tmp, _ = relu_forward(out_tmp)
#          #print(out_tmp.shape)
#          out[:,:,i,j] = out_tmp
#          #print(out.shape)
  
  for n in range(N):      
      for i in range(0,H_prime,1):
          for j in range(0,W_prime,1):
              h_strt = i*s 
              h_stp = h_strt + HH
              w_strt = j*s
              w_stp = w_strt + WW
              x_tmp = x_pad[n,:,h_strt:h_stp,w_strt:w_stp]
              for f in range(F):
                  out_tmp = np.sum(x_tmp * w[f,:,:,:]) + b[f] #one can prove that this is same as a dot product
                  out[n,f,i,j] = out_tmp

          
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  x,w,b,conv_param = cache

  s = conv_param['stride']
  p = conv_param['pad']
  
  x_pad = np.lib.pad(x,((0,0),(0,0),(p,p),(p,p)),'constant',constant_values=(0,0))

  #input shape
  N = x.shape[0]
  C = x.shape[1]
  H = x.shape[2]
  W = x.shape[3]

  #filter shape
  F = w.shape[0]
  #C is same as above
  HH = w.shape[2]
  WW = w.shape[3]
  w_reshaped = np.reshape(w,(-1,F))
  #no need to reshape b as it is of shape (F,)
  
  #output shape
  Hout = dout.shape[2]
  Wout = dout.shape[3]

  #initialize the gradients
  dx_pad = np.zeros_like(x_pad)
  dw = np.zeros_like(w)
  db = np.zeros_like(b)
  
  for n in range(N):
      for f in range(F):
          for ii in range(Hout):
              for jj in range(Wout):
                  strt_ii = ii*s
                  stp_ii = strt_ii+HH
                  strt_jj = jj*s
                  stp_jj = strt_jj+WW
                  
                  dw[f,:,:,:] += x_pad[n,:,strt_ii:stp_ii,strt_jj:stp_jj] * dout[n,f,ii,jj]
                  
                  dx_pad[n,:,strt_ii:stp_ii,strt_jj:stp_jj] += w[f,:,:,:] * dout[n,f,ii,jj]
                  
                  db[f] += dout[n,f,ii,jj]
                  
#  #Alternate implementation works too               
#  for f in range(F):
#      db[f] = np.sum(dout[:,f,:,:])
#  #db = np.sum(np.reshape(dout,(F,-1)), axis=1) #this doesn't work. I think there's soemthng weird with how reshape is done
  dx = dx_pad[:,:,p:p+H,p:p+W]
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  N, C, H, W = x.shape
  pool_h = pool_param['pool_height']
  pool_w = pool_param['pool_width']
  s = pool_param['stride']
  HO = int((H-pool_h)/s+1)
  WO = int((W-pool_w)/s+1)
  out_shape = (N, C, HO, WO)
  out = np.zeros(out_shape)
  
  for n in range(N):
      for c in range(C):
          for ii in range(HO):
              for jj in range(WO):
                  out[n,c,ii,jj] = np.max(x[n,c,s*ii:s*ii+pool_h,s*jj:s*jj+pool_w])
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  x,pool_param = cache
  N, C, H, W = x.shape
  pool_h = pool_param['pool_height']
  pool_w = pool_param['pool_width']
  s = pool_param['stride']
  HO = int((H-pool_h)/s+1)
  WO = int((W-pool_w)/s+1)
  
  dx = np.zeros_like(x)
  
  for n in range(N):
      for c in range(C):
          for ii in range(HO):
              for jj in range(WO):
                  tmp = x[n,c,s*ii:s*ii+pool_h,s*jj:s*jj+pool_w]
                  tmp2 = np.nonzero((tmp-(np.max(tmp))>=0)*1)
                  ii_max = tmp2[0][0]+s*ii
                  jj_max = tmp2[1][0]+s*jj
                  dx[n,c,ii_max,jj_max]=dout[n,c,ii,jj]
                  #print(n,c,ii_max,jj_max)
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  N,C,H,W = x.shape
  x0 = np.reshape(x,(N*H*W,C),order='C')
  x0_bn, cache = batchnorm_forward(x0,gamma,beta,bn_param)
  out = np.reshape(x0_bn,(N,C,H,W),order='C')
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  N,C,H,W = dout.shape
  dout0 = np.reshape(dout,(N*H*W,C),order='C')
  dx0,dgamma,dbeta = batchnorm_backward(dout0,cache)
  dx = np.reshape(dx0,(N,C,H,W),order='C')
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta
  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
