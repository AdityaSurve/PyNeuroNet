import tensorflow as tf
print(tf.version)

string=tf.Variable("this is a string",tf.string)
number=tf.Variable(324,tf.int16)
floating=tf.Variable(3.567,tf.float64)

print(string)
print(number)
print(floating)

rank1_tensor=tf.Variable(["Test","Python","Aditya"],tf.string)
rank2_tensor=tf.Variable([["Test1","Python","Aditya"],["Test2","Tensorflow","Surve"]],tf.string)

print(tf.rank(rank1_tensor)) 
print(tf.rank(rank2_tensor))
print(rank1_tensor.shape)
print(rank2_tensor.shape)

tensor_1=tf.ones([2,2,3])
print(tensor_1)
tensor_2=tf.reshape(tensor_1,[3,4])
print(tensor_2)
tensor_3=tf.reshape(tensor_1,[2,-1])
print(tensor_3)