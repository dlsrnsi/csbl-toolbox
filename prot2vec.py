#!/usr/bin/python

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import math
import random
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np

from gramizer import gramize, flatten_list


class Prot2Vec(object):
    """
    Prot2Vec model (Skipgram)

    Args:
        num_skips (int): How many words to consider left and right.
        skip_window (int): How many times to reuse an input to generate a label.
        batch_size (int): batch size to feed

    """

    def __init__(self, skip_window, num_skips, batch_size, normalized=True):
        self._skip_window = skip_window
        self._num_skips = num_skips
        self._batch_size = batch_size
        self._seq_index = 0
        self._gram_index = 0
        self._data = None
        self._count = None
        self._dictionary = None
        self._reverse_dictionary = None
        # We pick a random validation set to sample nearest neighbors. Here we limit the
        # validation samples to the words that have a low numeric ID, which by
        # construction are also the most frequent.
        self._valid_size = 16     # Random set of words to evaluate similarity on.
        self._valid_window = 100  # Only pick dev samples in the head of the distribution.
        self._valid_examples = np.random.choice(self._valid_window, self._valid_size, replace=False)
        self._num_sampled = 64    # Number of negative examples to sample.
        self._num_steps = 100001
        self._normalized = normalized

    def _build_dataset(self, seqs):
        """
        build dataset from gramized sequences and n-grams
        :param seqs: the list of gramized sequences
        :return data: the list of sequences which include indexes of word (instead of word)
        :return count: ("n-gram", count) pairs
        :return dictionary: ("n-gram", index) pairs
        :return reverse_dictionary: (index, "n-gram") pairs
        """
        grams = flatten_list(seqs)
        count = []
        count.extend(collections.Counter(grams).most_common())
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        data = list()
        for seq in seqs:
            seq_words = []
            for word in seq:
                index = dictionary[word]
                seq_words.append(index)
            data.append(seq_words)
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return data, count, dictionary, reverse_dictionary

    def build_dataset_from_seqlist(self, seq_list, n):
        """

        Prams:
            seq_list: iterable list of sequences
            n: specify the number of gram
        Return:
            count: list of pair ('word', count)
            dictionary: dictionary of word-index pair, {"word", index}
            reverse_dictionary: dictionary of index-word pair, {index, "word"}
        """
        grams = list(map(lambda seq: gramize(seq, n), seq_list))
        seqs = flatten_list(grams)

        print("whole sequences are gramized...")

        # Step 2, build dataset from parsed sequence

        self._data, self._count, self._dictionary, self._reverse_dictionary = self._build_dataset(seqs)
        del seqs
        del grams  # Hint to reduce memory.
        print('Most common words ', self._count[:5])
        print('Sample data', self._data[0][:10], [self._reverse_dictionary[i] for i in self._data[0][:10]])
        self.vocabulary_size = len(self._count)
        print("dataset from grammed ", str(len(self._data))," sequences is built")
        return self._count, self._dictionary, self._reverse_dictionary

# Step 3: Function to generate a training batch for the skip-gram model.
    def _generate_batch(self):
        """generate batch from original data with batch size for training. It will return batch and following label with skipping

        Args:
            batch_size (int): Dimension of the embedding vector.
            num_skips (int): How many words to consider left and right.
            skip_window (int): How many times to reuse an input to generate a label.
        Return:
            batch: numpy ndarray with [batch_size] dimension which is input gram
            labels: numpy ndarray with [batch_size,1] dimension which is yielded word with skipping
        """
        assert self._batch_size % self._num_skips == 0
        assert self._num_skips <= 2 * self._skip_window
        batch = np.ndarray(shape=(self._batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(self._batch_size, 1), dtype=np.int32)
        n_batch = 0

        def get_label_candidates(cur_index, num_skips, skip_window, len_seq):
            candidates = range(cur_index-skip_window, cur_index+skip_window)
            candidates = [candidate for candidate in candidates if (candidate>=0)&(candidate<=(len_seq-1))&(candidate!=cur_index)]
            sel_candidates = []
            while candidates:
                if len(sel_candidates) == num_skips:
                    break
                random.shuffle(candidates)
                sel_candidates.append(candidates.pop())
            return sel_candidates

        while n_batch != self._batch_size:
            candidates = get_label_candidates(self._gram_index, self._num_skips, self._skip_window, len(self._data[self._seq_index]))
            for candidate in candidates:
                if n_batch == self._batch_size: break
                batch[n_batch] = self._data[self._seq_index][self._gram_index]
                labels[n_batch] = self._data[self._seq_index][candidate]
                n_batch += 1
            self._gram_index = self._gram_index + 1
            if self._gram_index == len(self._data[self._seq_index]):
                self._gram_index = 0
                self._seq_index = (self._seq_index + 1)%len(self._data)
        return batch, labels

    def learn(self, embedding_size):
        """
        Training skip-gram model from data and return final embedding

        Args:
            embedding_size: the dimension of result word vector
        """
        self.graph = tf.Graph()

        with self.graph.as_default():

            # Input data.
            train_inputs = tf.placeholder(tf.int32, shape=[self._batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[self._batch_size, 1])
            valid_dataset = tf.constant(self._valid_examples, dtype=tf.int32)

            # Ops and variables pinned to the CPU because of missing GPU implementation
            with tf.device('/gpu:0'):
            # Look up embeddings for inputs.
                embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, embedding_size], -1.0, 1.0))
                embed = tf.nn.embedding_lookup(embeddings, train_inputs)

                # Construct the variables for the NCE loss
                nce_weights = tf.Variable(
                    tf.truncated_normal([self.vocabulary_size, embedding_size],
                                        stddev=1.0 / math.sqrt(embedding_size)))
                nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.
            loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=nce_weights,
                         biases=nce_biases,
                         labels=train_labels,
                         inputs=embed,
                         num_sampled=self._num_sampled,
                         num_classes=self.vocabulary_size))

            # Construct the SGD optimizer using a learning rate of 1.0.
            optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

            # Compute the cosine similarity between minibatch examples and all embeddings.
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(
                normalized_embeddings, valid_dataset)
            similarity = tf.matmul(
                valid_embeddings, normalized_embeddings, transpose_b=True)

            # Add variable initializer.
            init = tf.global_variables_initializer()

        # Step 5: Begin training.
        with tf.Session(graph=self.graph) as session:
            # We must initialize all variables before we use them.
            init.run()
            print("Initialized")

            average_loss = 0
            for step in xrange(self._num_steps):
                batch_inputs, batch_labels = self._generate_batch()
                feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

                # We perform one update step by evaluating the optimizer op (including it
                # in the list of returned values for session.run()
                _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
                average_loss += loss_val

                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                        # The average loss is an estimate of the loss over the last 2000 batches.
                        print("Average loss at step ", step, ": ", average_loss)
                        average_loss = 0

                # Note that this is expensive (~20% slowdown if computed every 500 steps)
                if step % 10000 == 0:
                    sim = similarity.eval()
                    for i in xrange(self._valid_size):
                        valid_word = self._reverse_dictionary[self._valid_examples[i]]
                        top_k = 8  # number of nearest neighbors
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log_str = "Nearest to %s:" % valid_word
                    for k in xrange(top_k):
                        close_word = self._reverse_dictionary[nearest[k]]
                        log_str = "%s %s," % (log_str, close_word)
                    print(log_str)
            if self._normalized:
                self._final_embeddings = normalized_embeddings.eval()
            else:
                self._final_embeddings = embeddings.eval()


    def n_gram2vec(self, n_gram):
        """
        vectorize one n-gram

        Args:
            n_gram: n-gram word
        Return:
            vectorized_gram: vectorized n-gram
        """
        vectorized_gram = self._final_embeddings[self._dictionary[n_gram],:]
        return vectorized_gram


    def n_grams2vec(self, n_grams):
        """
        vectorize many n-grams and return average of vector

        Args:
            n_grams: list of n-gram words
        Return:
            vectorized_grams: average of vectorized n-gram words
        """
        vectorized_grams = sum(list(map(lambda gram: self.n_gram2vec(gram), n_grams)))
        return vectorized_grams




if __name__ == "__main__":
    # step 1 : parse sequence data from input file
    import pandas as pd
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("n", help="the number of gram", default=333, type=int)
    parser.add_argument("input_file", help="input file to gramize")
    parser.add_argument("target_file", help="target sequence file to be vectorized")
    parser.add_argument("--batch-size", help="batch size for training", default=128, type=int)
    parser.add_argument("--embedding-size", help="dimension of vector", default=128, type=int)
    parser.add_argument("--skip-window", help="window size, How many words to consider left and right.", default=1, type=int)
    parser.add_argument("--num-skips", help="How many times to reuse an input to generate a label.", default=2, type=int)
    parser.add_argument("--output-file", help="output file")
    parser.add_argument("--index", help="use if you have index")
    parser.add_argument("--header", action="store_true", help="if file has header, use this argument")
    args = parser.parse_args()
    input_file = args.input_file
    target_file = args.target_file
    skip_window = args.skip_window
    num_skips = args.num_skips
    batch_size = args.batch_size
    embedding_size = args.embedding_size

    prot2vec = Prot2Vec(skip_window, num_skips, batch_size)

    if args.output_file:
        output_file = args.output_file
    else:
        output_file = args.target_file
    n = int(args.n)
    f = open(input_file)
    seq_list = f.readlines()
    f.close()

    count, dictionary, reverse_dictionary = prot2vec.build_dataset_from_seqlist(seq_list, n)
    final_embeddings = prot2vec.learn(embedding_size)

    '''
    # Step 6: Visualize the embeddings.


    def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
      assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
      plt.figure(figsize=(18, 18))  # in inches
      for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

      plt.savefig(filename)

    try:
      from sklearn.manifold import TSNE
      import matplotlib.pyplot as plt

      tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
      plot_only = 500
      low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
      labels = [reverse_dictionary[i] for i in xrange(plot_only)]
      plot_with_labels(low_dim_embs, labels)

    except ImportError:
      print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")
    '''

    # Step 6: Load target squences, gramize and vectorize.

    def n_grams2vec(n_grams_list):
        vectorized = np.zeros((final_embeddings.shape[1]), dtype=np.float32)
        for n_grams in n_grams_list:
            vec = sum(list(map(lambda gram: final_embeddings[dictionary[gram],:], n_grams)))
            vectorized.append(list(vec))
        return vectorized

    index = args.index

    if index:
        df = None
        if args.header:
            if target_file.endswith(".csv"):
                df = pd.read_csv(target_file)
            else:
                df = pd.read_table(target_file)
        else:
           if target_file.endswith(".csv"):
                df = pd.read_csv(target_file, header=None)
           else:
                df = pd.read_table(target_file, header=None)
        df[str(n)+"_gram"] = df[index].map(lambda seq: gramize(seq, n))
        df.dropna(inplace=True)
        df["vectorized"] = df[str(n)+"_gram"].map(n_grams2vec)
        df.to_csv(output_file)
        print("save to "+output_file)
    else:
        f = open(target_file)
        seq_list = f.readlines()
        f.close()
        grams = list(map(lambda seq: gramize(seq, n), seq_list))
        vectorized = list(map(n_grams2vec, grams))
