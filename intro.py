import tensorflow as tf

z = tf.zeros([10])
print(z) # Tensor o dimenzii 10

g = tf.get_default_graph()
print(g.as_graph_def())

tf.zeros([10], dtype=tf.int8) # Typovany tensor

# Session
s = tf.Session()
print(s.run(
	[], # Co chcem spravit
	{} # Feed dictionary 
))

print(z.eval(session=s)) # Plus mame moznost default sessions.

s = tf.InteractiveSession() # Interactive session - debugovanie, defaultna session je to hned po konstrukcii
print(z.eval())

# Premenne a trenovanie

# Definicia premennej - matica 10x10
v = tf.Variable(tf.random_normal([10, 10]))
print(v)

init = tf.initialize_all_variables() # Tensor, ktory ked sa evaluuje, tak do sessny inicializuje tieto premenne
print(init.run())

ones = tf.Variable(tf.ones([10]))
# ones.eval() # Neinicializovana

# Graph obsahuje:
#  - tenzory
#  - kolekcie
#    + summaries -> vysledky


