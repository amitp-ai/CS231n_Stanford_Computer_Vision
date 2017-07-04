import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  #for notational simplicity we write dLdW as dW
  #We compute the gradients for each training example and then take their average across the batch before updating the weights
  dW = np.zeros(W.shape) # initialize the gradient as zero
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]

  
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W) #scores is 1 X C dimensional vector
    correct_class_score = scores[y[i]]

    for j in range(num_classes):
      if j == y[i]:
        continue
      
      else:
        margin = scores[j] - correct_class_score + 1 # note delta = 1
        
      if margin > 0: #b'cse SVM loss has a max function inside it, gradient will be zero if margin <= 0
        loss += margin
        dW[:,j] = dW[:,j] + X[i,:] #sums over all the training examples for a given j
        dW[:,y[i]] = dW[:,y[i]] + -X[i,:] #sums over all j!=y[i] and all the training examples
        


  # Right now the loss/gradient is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train  #take the average loss across all the examples
  dW /= num_train #take the average gradient across all the examples

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather than first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_examples = X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = np.dot(X,W)
  correct_class_score = scores[np.arange(num_examples),y]
  correct_class_score = np.reshape(correct_class_score,
                                   (correct_class_score.shape[0],1))
  mat = scores - correct_class_score + 1
  max_mask = mat<=0
  mat[max_mask]=0
  mat[np.arange(num_examples),y] = 0 #implementing the max function
      
  loss = np.sum(mat,axis=1)
  loss = np.mean(loss)
  loss += 0.5*reg*np.sum(W*W)
      
  dLdS = np.ones(scores.shape).astype(np.float64)
  dLdS[max_mask] = 0.0
  tmp = np.sum(dLdS,axis=1)
  dLdS[np.arange(num_examples),y] = -1.0*tmp #tmp[np.arange(num_examples)]
  dSdW = X
  dW = np.dot(dSdW.T,dLdS)
  dW/=num_examples
  dW += reg*W
      
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
