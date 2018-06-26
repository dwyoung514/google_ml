import tensorflow as tf
import matplotlib.pyplot as plt # Dataset visualization.
import numpy as np              # Low-level numerical Python library
import pandas as pd             # Higher-level numerical Python library.
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

c = tf.constant('Hello, world!')

#with tf.Session() as sees:
#    print (sees.run(c))


#x = tf.constant(5.2)

#y = tf.Variable([5])

#y = y.assign([3])

#with tf.Session() as sees:
#    initialization = tf.global_variables_initializer()
#    print (y.eval())


# Create a graph.
g = tf.Graph()

# Establish the graph as the "default" graph.
with g.as_default():
  # Assemble a graph consisting of the following three operations:
  #   * Two tf.constant operations to create the operands.
  #   * One tf.add operation to add the two operands.
  x = tf.constant(8, name="x_const")
  y = tf.constant(5, name="y_const")
  z = tf.constant(4, name="z_const")
  my_sum = tf.add(x, y, name="x_y_sum")
  new_sum = tf.add(my_sum, z, name="x_y_z_sum")

  # Now create a session.
  # The session will run the default graph.
  with tf.Session() as sess:
    print (new_sum.eval())



#For this section, you can use tensors to add multiple matrices that are
#identical in size and shape
with tf.Graph().as_default():
  # Create a six-element vector (1-D tensor).
  primes = tf.constant([2, 3, 5, 7, 11, 13], dtype=tf.int32)

  # Create another six-element vector. Each element in the vector will be
  # initialized to 1. The first argument is the shape of the tensor (more
  # on shapes below).
  ones = tf.ones([6], dtype=tf.int32)

  # Add the two vectors. The resulting tensor is a six-element vector.
  just_beyond_primes = tf.add(primes, ones)

  # Create a session to run the default graph.
  with tf.Session() as sess:
    print (just_beyond_primes.eval())

with tf.Graph().as_default():
  # A scalar (0-D tensor).
  scalar = tf.zeros([])

  # A vector with 3 elements.
  vector = tf.zeros([3])

  # A matrix with 2 rows and 3 columns.
  matrix = tf.zeros([2, 3])

  with tf.Session() as sess:
    print ('scalar has shape', scalar.get_shape(), 'and value:\n',
            scalar.eval())
    print ('vector has shape', vector.get_shape(), 'and value:\n',
            vector.eval())
    print ('matrix has shape', matrix.get_shape(), 'and value:\n',
    matrix.eval())


#Broadcasting allows the use of smaller tensors to do operations
with tf.Graph().as_default():
  # Create a six-element vector (1-D tensor).
  primes = tf.constant([2, 3, 5, 7, 11, 13], dtype=tf.int32)

  # Create a constant scalar with value 1.
  ones = tf.constant(1, dtype=tf.int32)

  # Add the two tensors. The resulting tensor is a six-element vector.
  just_beyond_primes = tf.add(primes, ones)

  with tf.Session() as sess:
    print (primes.eval())
    print(ones.eval())
    print (just_beyond_primes.eval())

#matrix multipliations. columns in first must equal rows of second
with tf.Graph().as_default():
  # Create a matrix (2-d tensor) with 3 rows and 4 columns.
  x = tf.constant([[5, 2, 4, 3], [5, 1, 6, -2], [-1, 3, -1, -2]],
                  dtype=tf.int32)

  # Create a matrix with 4 rows and 2 columns.
  y = tf.constant([[2, 2], [3, 5], [4, 5], [1, 6]], dtype=tf.int32)

  # Multiply `x` by `y`.
  # The resulting matrix will have 3 rows and 2 columns.
  matrix_multiply_result = tf.matmul(x, y)

  with tf.Session() as sess:
    print (matrix_multiply_result.eval())


#Tensor Reshaping
#It appears that values are taken in order as they appear from by row left to
#right and column top to bottom and placed in the new matrix in the order they
#were taken
with tf.Graph().as_default():
  # Create an 8x2 matrix (2-D tensor).
  matrix = tf.constant([[1,2], [3,4], [5,6], [7,8],
                        [9,10], [11,12], [13, 14], [15,16]], dtype=tf.int32)

  # Reshape the 8x2 matrix into a 2x8 matrix.
  reshaped_2x8_matrix = tf.reshape(matrix, [2,8])

  # Reshape the 8x2 matrix into a 4x4 matrix
  reshaped_4x4_matrix = tf.reshape(matrix, [4,4])

  with tf.Session() as sess:
    print ("Original matrix (8x2):")
    print (matrix.eval())
    print ("Reshaped matrix (2x8):")
    print (reshaped_2x8_matrix.eval())
    print ("Reshaped matrix (4x4):")
    print (reshaped_4x4_matrix.eval())

with tf.Graph().as_default():
  # Create an 8x2 matrix (2-D tensor).
  matrix = tf.constant([[1,2], [3,4], [5,6], [7,8],
                        [9,10], [11,12], [13, 14], [15,16]], dtype=tf.int32)

  # Reshape the 8x2 matrix into a 3-D 2x2x4 tensor.
  reshaped_2x2x4_tensor = tf.reshape(matrix, [2,2,4])
  
  # Reshape the 8x2 matrix into a 1-D 16-element tensor.
  one_dimensional_vector = tf.reshape(matrix, [16])

  with tf.Session() as sess:
    print ("Original matrix (8x2):")
    print (matrix.eval())
    print ("Reshaped 3-D tensor (2x2x4):")
    print (reshaped_2x2x4_tensor.eval())
    print ("1-D vector:")
    print (one_dimensional_vector.eval())

###
#Exercise #1: Reshape two tensors in order to multiply them.
#The following two vectors are incompatible for matrix multiplication:

#a = tf.constant([5, 3, 2, 7, 1, 4])
#b = tf.constant([4, 6, 3])
#Reshape these vectors into compatible operands for matrix multiplication. Then, invoke a matrix multiplication operation on the reshaped tensors.
with tf.Graph().as_default():
  #1x6 -> 2x3
  a = tf.constant([5, 3, 2, 7, 1, 4])
  #1x3 -> 3x1
  b = tf.constant([4, 6, 3])

  reshaped_a = tf.reshape(a, [2,3])
  reshaped_b = tf.reshape(b, [3,1])
  a_b_multiply = tf.matmul(reshaped_a,reshaped_b)
  with tf.Session() as sess:
    print ("original a (1x6):")
    print (a.eval())
    print ("reshaped a (2x3):")
    print (reshaped_a.eval())
    print ("original b (1x3):")
    print (b.eval())
    print ("reshaped b (3x1):")
    print (reshaped_b.eval())
    print ("reshaped a times reshaped b:")
    print (a_b_multiply.eval())

#Creating global variables
g = tf.Graph()
with g.as_default():
  # Create a variable with the initial value 3.
  v = tf.Variable([3])

  # Create a variable of shape [1], with a random initial value,
  # sampled from a normal distribution with mean 1 and standard deviation 0.35.
  w = tf.Variable(tf.random_normal([1], mean=1.0, stddev=0.35))

with g.as_default():
  with tf.Session() as sess:
    initialization = tf.global_variables_initializer()
    sess.run(initialization)
    # Now, variables can be accessed normally, and have values assigned to them.
    print (v.eval())
    print (w.eval())

with g.as_default():
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # This should print the variable's initial value.
    print (v.eval())

    assignment = tf.assign(v, [7])
    # The variable has not been changed yet!
    print (v.eval())

    # Execute the assignment op.
    sess.run(assignment)
    # Now the variable is updated.
    print (v.eval())


#Exercise #2: Simulate 10 rolls of two dice.
#Create a dice simulation, which generates a 10x3 2-D tensor in which:

#Columns 1 and 2 each hold one throw of one six-sided die (with values 1–6).
#Column 3 holds the sum of Columns 1 and 2 on the same row.
#For example, the first row might have the following values:

#Column 1 holds 4
#Column 2 holds 3
#Column 3 holds 7
#You'll need to explore the TensorFlow documentation to solve this task.

g = tf.Graph()
with g.as_default():
  dice1 = tf.Variable(tf.random_uniform([10,1], minval=1, maxval=7, dtype=tf.int32))
  dice2 = tf.Variable(tf.random_uniform([10,1], minval=1, maxval=7, dtype=tf.int32))


with g.as_default():
  with tf.Session() as sess:
    initialization = tf.global_variables_initializer()
    sess.run(initialization)

    #print (dice1.eval())
    #print (dice2.eval())

    dice_sum = tf.add(dice1,dice2)
    #print (dice_sum.eval())

    dice_matrix = tf.concat(values=[dice1,dice2,dice_sum], axis=1)

    print (dice_matrix.eval())

