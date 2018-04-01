import numpy as np
from collections import Counter
from cs231n.data_utils import load_CIFAR10
Xtr, Ytr, Xte, Yte = load_CIFAR10('C:/Users/dk12a7/Downloads/Compressed/cifar-10-batches-py/') # a magic function we provide
# flatten out all images to be one-dimensional
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xte_rows becomes 10000 x 3072



class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y

  def predict(self, X,k):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

    # loop over all test rows
    for i in xrange(num_test):
      print i
      # find the nearest training image to the i'th test image
      # using the L1 distance (sum of absolute value differences)
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)      
      min_indexes = np.argpartition(distances, k)[:k] # get the index with smallest distance
      yget = self.ytr[min_indexes]
      c = Counter(yget).most_common(k)
      result = c[0][0]
      if c[0][1]==1:
        k = 1
        min_indexes = np.argpartition(distances, k)[:k]
        yget = self.ytr[min_indexes]
        c = Counter(yget).most_common(k)
        result = c[0][0]
      else:
        i = 0
        while i<len(c)-1 and c[i][1]==c[i+1][1]:
          i = i+1
        if i!=0:            
          result = c[0][0]
          k2 = 1
          ind2 = np.argpartition(distances, k2)[:k2]
          yget2 = self.ytr[ind2]
          c2 = Counter(yget2).most_common(1)
          for y in range(0,i):
            if c[i][0]==c2[0][0]:
              result = c[i][0]
      Ypred[i] = result # predict the label of the nearest example

    return Ypred

nn = NearestNeighbor() # create a Nearest Neighbor classifier class
nn.train(Xtr_rows, Ytr) # train the classifier on the training images and labels
Yte_predict = nn.predict(Xte_rows,7) # predict labels on the test images
# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)
print 'accuracy: %f' % ( np.mean(Yte_predict == Yte) )