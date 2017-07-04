import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_examples = X.shape[0]
  num_features = X.shape[1]
  num_classes = W.shape[1]
  
  scores = np.dot(X,W)
  loss = 0.0
  for i in range(num_examples):
      
      scores[i] = scores[i]-np.max(scores[i]) #for numerical stability. See http://cs231n.github.io/linear-classify/#softmax
      correct_class_scores = scores[i,y[i]]
      SM = np.exp(correct_class_scores)/np.sum(np.exp(scores[i]))
      loss += -np.log(SM)
      
      temp1 = np.exp(scores[i])/np.sum(np.exp(scores[i]))
      temp1[y[i]] = SM-1
      temp1 = np.reshape(temp1,(1,num_classes))
      temp2 = np.reshape(X[i],(num_features,1))
      dW += np.dot(temp2,temp1)
    


  loss /= num_examples
  loss += 0.5*reg*np.sum(W*W)
  
  dW /= num_examples
  dW += reg*W

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_examples = X.shape[0]
  num_classes = W.shape[1]
  
  scores = np.dot(X,W)
  scores_max = np.max(scores,axis=1)
  scores_max = np.reshape(scores_max,(num_examples,1))
  scores = scores - scores_max #for numerical stability. See http://cs231n.github.io/linear-classify/#softmax
  
  exp_sum = np.sum(np.exp(scores),axis=1)
  exp_sum = np.reshape(exp_sum,(num_examples,1))

  correct_class_scores = scores[np.arange(num_examples),y]
  correct_class_scores = np.reshape(correct_class_scores,(num_examples,1))
  SM = np.exp(correct_class_scores)/exp_sum
  
  temp1 = np.exp(scores)/exp_sum
  temp1[np.arange(num_examples),y] = np.reshape(SM-1,(num_examples,))
  dW = np.dot(X.T,temp1)
  dW /= num_examples
  dW += reg*W
  
  
  loss = -np.log(SM)
  loss = np.mean(loss)
  loss += 0.5*reg*np.sum(W*W)
  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

