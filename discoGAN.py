from __future__ import print_function

import tensorflow as tf
import numpy as np



def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def huber_loss(logits,labels,max_gradient=1.0, name="huber_loss"):
    with tf.variable_scope(name):
        err = tf.abs(labels-logits)
        mg = tf.constant(max_gradient)
        lin = mg*(err-0.5*mg)
        quad = 0.5*err*err
        return tf.where(err<mg,quad,lin)

class DiscoGAN(object):

    def __init__(self, batch_size):
        self._batch_size = batch_size
        self._n_train = 10000
        self._index = 0
        self._data_size = 0
        self._A_size = 0
        self._B_size = 0
        self._conv = False

    def _generator(self, input_value, output_size, hidden_size, n_hiddens, reuse=False, name=None, eval=None, sig=False):
        """
        Feed network from input to hidden layers

        Args:
            hiddens:
            biases:
        Return:
            Generated output from network
        """
        with tf.variable_scope("generator_"+name):
            prev_value = input_value
            for i in range(n_hiddens):
                layer = tf.contrib.layers.fully_connected(inputs=prev_value, num_outputs=hidden_size, reuse=reuse,
                                                         normalizer_fn=tf.contrib.layers.batch_norm, activation_fn = tf.nn.relu,
                                                         weights_initializer=self._initializer, scope="g_fc_"+name+"_"+str(i+1))
                prev_value = layer
            if sig:
                output_value = tf.contrib.layers.fully_connected(inputs=prev_value, num_outputs=output_size,
                                                         activation_fn = tf.nn.sigmoid, normalizer_fn=tf.contrib.layers.batch_norm,
                                                         weights_initializer=self._initializer, scope="g_fc_"+name+"_"+"output")
            else:
                output_value = tf.contrib.layers.fully_connected(inputs=prev_value, num_outputs=output_size,
                                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                                         weights_initializer=self._initializer, scope="g_fc_"+name+"_"+"output")
            if eval:
                return output_value.eval(session=eval)
            else:
                return output_value

    def _discriminate(self, input_value, hidden_size, n_hiddens, reuse=False, name=None):
        """
        Feed network from input to hidden layers

        Args:
            hiddens:
            biases:
        Return:
            Generated output from network
        """
        with tf.variable_scope("discriminator_"+name):
            prev_value = input_value
            for i in range(n_hiddens):
                layer = tf.contrib.layers.fully_connected(inputs=prev_value, num_outputs=hidden_size, reuse=reuse,
                                                         activation_fn = lrelu, normalizer_fn=tf.contrib.layers.batch_norm,
                                                         weights_initializer=self._initializer, scope="d_fc_"+name+"_"+str(i+1))
                prev_value = layer
            output_value = tf.contrib.layers.fully_connected(inputs=prev_value, num_outputs=1,
                                                         activation_fn = tf.nn.sigmoid, normalizer_fn=tf.contrib.layers.batch_norm,
                                                         weights_initializer=self._initializer, scope="d_fc_"+name+"_"+"output")
            return output_value

    def _generate_batch(self):
        batch_A = []
        batch_B = []
        n = 0
        while n!=self._batch_size:
            batch_A.append(list(self._A[self._index]))
            batch_B.append(list(self._B[self._index]))
            self._index = (self._index + 1) % self._data_size
            n +=1
        batch_A = np.array(batch_A, dtype=np.float32)
        batch_B = np.array(batch_B, dtype=np.float32)
        return batch_A, batch_B

    def learn(self, A, B, hidden_size=64, n_hidden=2, learning_rate=0.01, n_train=10000, conv=False):
        """
        Train model with input data

        :param A: set of A domain data
        :param B: set of B domain data
        :param hidden_size: dimension size of hidden layer
        :param n_hidden: the number of hidden layer
        :param learning_rate: Learning rate
        :param n_train: the number of train
        :return:
        """
        A = np.asarray(A, dtype=np.float32)
        B = np.asarray(B, dtype=np.float32)
        self._A = A
        self._B = B
        self._data_size = A.shape[0]
        self._A_size = A.shape[1]
        self._B_size = B.shape[1]
        self._n_train = n_train
        self._graph = tf.Graph()
        self._hidden_size = hidden_size
        self._n_hidden = n_hidden
        self._learning_rate = learning_rate
        self._initializer = tf.truncated_normal_initializer(stddev=1)
        self._conv = conv

        with self._graph.as_default():
            with tf.device("/gpu:0"):
                A_size = self._A_size
                B_size = self._B_size
                A = tf.placeholder(tf.float32, [None, A_size], name="A")
                B = tf.placeholder(tf.float32, [None, B_size], name="B")
                AB = self._generator(A, B_size, self._hidden_size, self._n_hidden, name="AB", sig=True)
                BA = self._generator(B, A_size, self._hidden_size, self._n_hidden, name="BA")
                ABA = self._generator(AB, A_size, self._hidden_size, self._n_hidden, name="BA", reuse=True)
                BAB = self._generator(BA, B_size, self._hidden_size, self._n_hidden, name="AB", reuse=True, sig=True)
                D_A_fake = self._discriminate(BA, self._hidden_size, self._n_hidden, name="A")
                D_B_fake = self._discriminate(AB, self._hidden_size, self._n_hidden, name="B")
                D_A_real = self._discriminate(A, self._hidden_size, self._n_hidden, name="A", reuse=True)
                D_B_real = self._discriminate(B, self._hidden_size, self._n_hidden, name="B", reuse=True)

                self._predicted_A = self._generator(B, A_size, self._hidden_size, self._n_hidden, name="BA", reuse=True)
                self._predicted_B = self._generator(A, B_size, self._hidden_size, self._n_hidden, name="AB", reuse=True)

                with tf.variable_scope("const_loss"):
                    const_A = tf.reduce_mean(tf.sqrt(tf.square(A-ABA)))
                    const_B = tf.reduce_mean(tf.sqrt(tf.square(B-BAB)))
                with tf.variable_scope("generator_loss"):
                    cost_G_A = tf.reduce_mean(tf.square(D_A_fake-1))
                    cost_G_B = tf.reduce_mean(tf.square(D_B_fake-1))
                with tf.variable_scope("discriminator_loss"):
                    cost_D_A = tf.reduce_mean(tf.square(D_A_real-1)) + tf.reduce_mean(tf.square(D_A_fake))
                    cost_D_B = tf.reduce_mean(tf.square(D_B_real-1)) + tf.reduce_mean(tf.square(D_B_fake))
                with tf.variable_scope("total_loss"):
                    cost_G = const_A + const_B + cost_G_A + cost_G_B
                    cost_D = cost_D_A + cost_D_B

                self._g_ab_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator_AB")
                self._g_ba_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator_BA")
                d_a_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator_A")
                d_b_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator_B")

                optimizer_G = tf.train.AdadeltaOptimizer(self._learning_rate)
                optimizer_D = tf.train.AdadeltaOptimizer(self._learning_rate)

                g_grads = optimizer_G.compute_gradients(cost_G, self._g_ab_var + self._g_ba_var)
                d_grads = optimizer_D.compute_gradients(cost_D, d_a_var + d_b_var)

                update_D = optimizer_G.apply_gradients(g_grads)
                update_G = optimizer_D.apply_gradients(d_grads)

            init = tf.global_variables_initializer()
            tf.summary.scalar("disc_loss", cost_D)
            tf.summary.scalar("gen_loss", cost_G)
            merged = tf.summary.merge_all()

        self._sess = tf.Session(graph=self._graph)
        writer = tf.summary.FileWriter("./log/nn_logs", self._sess.graph)
        self._sess.run(init)

        for i in range(self._n_train):
            feed_A, feed_B = self._generate_batch()
            feed_dict = {A:feed_A, B:feed_B}
            c_1,c_2, _, _, summary = self._sess.run([cost_G, cost_D, update_D, update_G, merged], feed_dict=feed_dict)
            writer.add_summary(summary, i)
            if i%500==0:
                print("Sum of root mean square loss :", str(c_1+c_2))
                print(" Loss of generation : " + str(c_1))
                print(" Loss of discrimination : " + str(c_2))

    def generate_foward(self, input_value):
        predicted = iter(self._g_ab_var)
        layer = input_value
        while True:
            try:
                weight = next(predicted)
                layer = tf.matmul(layer, weight)
                bias = next(predicted)
                layer = layer + bias
            except StopIteration:
                layer = tf.nn.sigmoid(layer)
                return layer.eval(session=self._sess)

    def generate_reverse(self, input_value):
        predicted = iter(self._g_ba_var)
        layer = input_value
        while True:
            try:
                weight = next(predicted)
                layer = tf.matmul(layer, weight)
                bias = next(predicted)
                layer = layer + bias
            except StopIteration:
                return layer.eval(session=self._sess)

    def reconstructA(self, input_value):
        fowarded = self.generate_foward(input_value)
        reconstructed = self.generate_reverse(fowarded)
        return reconstructed

    def reconstructB(self, input_value):
        reversed = self.generate_reverse(input_value)
        reconstructed = self.generate_reverse(reversed)
        return reconstructed
