from __future__ import print_function, division
from builtins import range
import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    next_h = np.tanh(np.dot(x,Wx) + np.dot(prev_h,Wh) + b)
    cache = (x,Wx,prev_h,Wh,next_h)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    dtanh = (1 - cache[4]*cache[4]) * dnext_h #N x H
    dx = np.dot(dtanh, cache[1].T) #N X D
    dprev_h = np.dot(dtanh, cache[3].T) #N X H
    dWx = np.dot(cache[0].T, dtanh) #D X H
    dWh = np.dot(cache[2].T, dtanh) #N X H
    db = dtanh.sum(0) #H X 1
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    N,T,D = x.shape
    _,H = h0.shape
    h = np.ones([N,T,H])
    hprev = h0
    cache = []
    for i in range(T):
        xtemp = x[:,i,:].reshape(N,D)
        hprev,ctemp = rnn_step_forward(xtemp, hprev, Wx, Wh, b)
        h[:,i,:] = hprev.reshape(N,H) #Actually no need to reshape hprev
        #print(hprev.shape)
        cache.append(ctemp)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H)

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    N,T,H = dh.shape
    _,D = cache[0][0].shape
    
    dx = np.zeros([N,T,D])
    dh0 = np.zeros([N,H])
    dWx = np.zeros([D,H])
    dWh = np.zeros([H,H])
    db = np.zeros(H)

    for i in range(len(cache)-1, -1, -1):
        
        dnext_h = dh[:,i,:]
        for j in range(i, -1, -1):

            dx_tmp,dnext_h,dWx_tmp,dWh_tmp,db_tmp = rnn_step_backward(dnext_h, cache[j])
            
            dx[:,j,:] += dx_tmp
            dWh += dWh_tmp
            dWx += dWx_tmp
            db += db_tmp
            
        
        dh0 += dnext_h #final value of dnext_h from inner loop
#        dnext_h = dh[:,i,:] + dh0
#        _, tmp_dprev_h, _, _, _ = rnn_step_backward(dnext_h, cache[i])          
#        dh0 = tmp_dprev_h        
        
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using NumPy's array indexing.           #
    ##############################################################################
    N,T = x.shape
    D = W.shape[1]
    out = np.zeros([N,T,D])
    for i in range(N):
        for j in range(T):
            out[i,j,:] = W[x[i,j],:]
    
    cache = (x, W) #need for backpropagation
            
            
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that Words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    x,W = cache
    dW = np.zeros_like(W)
    N,T = x.shape
    
    for i in range(N):
        for j in range(T):
            dW[x[i,j],:] += dout[i,j,:]
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    _,H = prev_h.shape
    tmp_full = np.dot(x,Wx) + np.dot(prev_h,Wh) + b #NX4H
    tmp_i = sigmoid(tmp_full[:,0:H]) #NXH
    tmp_f = sigmoid(tmp_full[:,H:2*H]) #NXH
    tmp_o = sigmoid(tmp_full[:,2*H:3*H]) #NXH
    tmp_g = np.tanh(tmp_full[:,3*H:4*H]) #NXH
    
    next_c = tmp_f * prev_c + tmp_i * tmp_g #elementwise product NXH
    next_h = tmp_o * np.tanh(next_c) #elementwise product NXH
    
    cache = (x, Wx, prev_h, Wh, tmp_i, tmp_full, next_c, prev_c)
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    x, Wx, prev_h, Wh, tmp_i, tmp_full, next_c, prev_c = cache[0],cache[1],cache[2],cache[3],cache[4],cache[5],cache[6],cache[7]
    _,H = prev_h.shape
    tmp_i = sigmoid(tmp_full[:,0:H]) #NXH
    tmp_f = sigmoid(tmp_full[:,H:2*H]) #NXH
    tmp_o = sigmoid(tmp_full[:,2*H:3*H]) #NXH
    tmp_g = np.tanh(tmp_full[:,3*H:4*H]) #NXH
    
                    
    #intermediate variables
    tanh_cnext = np.tanh(next_c)
    dl_h_dc = dnext_h * tmp_o * (1-tanh_cnext*tanh_cnext) #NXH
    gamma = dnext_c + dl_h_dc #NXH  

    dhdo = tanh_cnext #NXH
    alpha = np.zeros_like(tmp_full) #NX4H
    alpha[:,2*H:3*H] = dnext_h * dhdo * (tmp_o*(1-tmp_o))#NX4H
    
    dcdf = prev_c #NXH
    dfdfull = np.zeros_like(tmp_full) #NX4H
    dfdfull[:,H:2*H] = gamma * dcdf * (tmp_f*(1-tmp_f))
    dcdi = tmp_g #NXH
    didfull = np.zeros_like(tmp_full) #NX4H
    didfull[:,0:H] = gamma * dcdi * (tmp_i*(1-tmp_i))
    dcdg = tmp_i #NXH
    dgdfull = np.zeros_like(tmp_full) #NX4H
    dgdfull[:,3*H:4*H] = gamma * dcdg * (1-tmp_g*tmp_g)
    beta = dfdfull + didfull + dgdfull #N*4H
    #
    
    #dprev_c
    dprev_c = gamma * tmp_f#NXH
    #dWx
    dWx = np.dot(x.T,alpha+beta)
    #dWh
    dWh = np.dot(prev_h.T,alpha+beta)
    #dx
    dx = np.dot(alpha+beta,Wx.T)
    #dprev_h
    dprev_h = np.dot(alpha+beta,Wh.T)
    #db
    db = np.sum(alpha+beta,axis=0)
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    #Note: Given cell is an internal variable and is not returned, there is no way for 
    #cell to directly influence the loss i.e. dLtdct = 0
    h_prev = h0
    c_prev = np.zeros_like(h0)
    
    N,T,D = x.shape
    N,H = h0.shape
    
    h = np.zeros([N,T,H])
    cache = []
    for i in range(T):
        xtmp = x[:,i,:].reshape(N,D)
        h_prev, c_prev, cache_tmp = lstm_step_forward(xtmp, h_prev, c_prev, Wx, Wh, b)
        cache.append(cache_tmp)
        h[:,i,:] = h_prev #don't have to reshape h_prev

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    N,T,H = dh.shape
    _,D = cache[0][0].shape #x
    
    dx = np.zeros([N,T,D])
    dh0 = np.zeros([N,H])
    dWx = np.zeros([D,4*H])
    dWh = np.zeros([H,4*H])
    db = np.zeros([4*H])
    
    #Note: Given cell is an internal variable and is not returned, there is no way for 
    #cell to directly influence the loss i.e. dLtdct = 0 for every time step
    
    for i in range(T-1,-1,-1):
        dc_tmp = np.zeros([N,H])
        dh_tmp = dh[:,i,:].reshape(N,H)
        
        for j in range(i,-1,-1):
            dx_tmp, dh_tmp, dc_tmp, dWx_tmp, dWh_tmp, db_tmp = lstm_step_backward(dh_tmp, dc_tmp, cache[j])
            
            dx[:,j,:] += dx_tmp
            dWx += dWx_tmp
            dWh += dWh_tmp
            db += db_tmp
            
    
        dh0 += dh_tmp #add only the last value of dh_tmp from inner loop
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
