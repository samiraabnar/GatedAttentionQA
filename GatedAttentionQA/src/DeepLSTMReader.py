from BaseReaderModel import BaseReaderModel
from DataReader import *
from utils import Util

import tensorflow as tf
import numpy as np
import time
import os
from glob import glob


class DeepLSTMReader(BaseReaderModel):
    def __init__(self,sess,hparams,mode,data_reader):
        super(DeepLSTMReader, self).__init__()

        self.model_name = "DeepLSTM"
        self.hparams = hparams
        self.mode = mode
        self.sess = sess
        self.data_reader = data_reader

        self.vocab, self.rev_vocab = self.data_reader.load_vocab(data_dir=hparams.data_dir, dataset_name=hparams.dataset_name, vocab_size=hparams.vocab_size)
        self.vocab_size = len(self.vocab.keys())

    def define_graph(self):

        filenames = glob(os.path.join("../data", "cnn_*.tfrecords"))
        min_after_dequeue = 1000

        filename_queue = tf.train.string_input_producer(
            filenames)
        document, question, answer, document_shape, question_shape, answer_shape = read_tf_record_file(filename_queue)

        d_batch, q_batch, ans_batch, document_shape_batch, question_shape_batch, answer_shape_batch = tf.train.shuffle_batch(
            [document, question, answer, document_shape, question_shape, answer_shape], batch_size=self.hparams.batch_size,
            capacity=min_after_dequeue * 3 + 1, min_after_dequeue=min_after_dequeue)
        d_q_batch = tf.sparse_concat(axis=1, sp_inputs=[d_batch, q_batch], )
        dense_d_q_batch = tf.sparse_to_dense(sparse_indices=d_q_batch.indices,
                                             output_shape=d_q_batch.dense_shape,
                                             sparse_values=d_q_batch.values,
                                             default_value=0,
                                             validate_indices=True,
                                             name=None)
        dens_ans_batch = tf.sparse_to_dense(sparse_indices=ans_batch.indices,
                                            output_shape=ans_batch.dense_shape,
                                            sparse_values=ans_batch.values,
                                            default_value=0,
                                            validate_indices=True,
                                            name=None)
        d_q_lengths = tf.reduce_sum(tf.concat(
            [tf.reshape(document_shape_batch, (self.hparams.batch_size, 1)), tf.reshape(question_shape_batch, (self.hparams.batch_size, 1))],
            axis=1), axis=1)




        self.embedding = tf.get_variable("embedding",[self.vocab_size,self.hparams.number_of_hidden_units],trainable=True)
        tf.logging.info(self.embedding.get_shape())
        self.inputs = dense_d_q_batch
        self.y = dens_ans_batch[:,0]#tf.placeholder(tf.float32, [self.hparams.self.hparams.batch_size, self.vocab_size])

        #unstacked_inputs = tf.unstack(self.inputs,axis=1)
        embedded_inputs = [tf.nn.embedding_lookup(self.embedding, self.inputs[i] ) for i in range(self.hparams.batch_size)]

        #embedded_inputs = tf.stack(embedded_inputs)
        tf.summary.histogram("embeddings",self.embedding)

        self.__build_deep_lstm_cell()
        states_series, current_state = tf.nn.dynamic_rnn(cell=self.stacked_cell,
                                                inputs=tf.stack(embedded_inputs),
                                                sequence_length=d_q_lengths,
                                                initial_state=None,
                                                dtype=tf.float32,
                                                parallel_iterations=None,
                                                swap_memory=False,
                                                time_major=False,
                                                scope=None)


        self.batch_states = [layer_state[1] for layer_state in tf.unstack(current_state)]# tf.stack(states)


        self.output_size = self.hparams.depth * self.hparams.number_of_hidden_units

        """startings = tf.concat([
                            tf.zeros((self.hparams.batch_size,1),dtype=tf.int64),
                            tf.reshape(d_q_lengths,(self.hparams.batch_size,1)) - 1,
                            tf.zeros((self.hparams.batch_size,1),dtype=tf.int64)],axis=1)
        outputs =  [self.batch_states[0][i,d_q_lengths[i],:] for i in range(self.hparams.batch_size)]"""

        outputs= tf.concat(self.batch_states,axis=1)
        tf.logging.set_verbosity(tf.logging.INFO)
        tf.logging.info(outputs)
        tf.logging.info(self.batch_states)
        tf.logging.info(current_state[0])



        self.outputs = tf.reshape(outputs, [self.hparams.batch_size, self.output_size])

        self.W = tf.get_variable("W", [self.output_size,self.vocab_size,],trainable=True)
        tf.summary.histogram("weights", self.W)
        tf.summary.histogram("output", self.outputs)

        self.y_ = tf.matmul(self.outputs,self.W)
        tf.logging.info(self.y_)
        cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.y_,labels=self.y)
        self.train_loss = (tf.reduce_sum(cross_ent) /
                      self.hparams.batch_size)
        tf.summary.scalar("loss", tf.reduce_mean(self.train_loss))
        tf.logging.info(tf.argmax(self.y_, 1))
        correct_prediction = tf.equal(self.y, tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        tf.summary.scalar("accuracy", self.accuracy)


    def __build_deep_lstm_cell(self):
        self.cell = tf.contrib.rnn.LSTMCell(self.hparams.number_of_hidden_units, forget_bias=0.0, state_is_tuple=True)

        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.cell = tf.nn.rnn_cell.DropoutWrapper(self.cell, output_keep_prob=self.hparams.keep_prob)
        #elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
        #else: #if self.mode == tf.contrib.learn.ModeKeys.INFER:
        self.stacked_cell = tf.contrib.rnn.MultiRNNCell([self.cell] * self.hparams.depth,state_is_tuple=True)

        self.initial_state = self.stacked_cell.zero_state(self.hparams.batch_size, tf.float32)


    def _define_train(self):
        warmup_steps = self.hparams.learning_rate_warmup_steps
        warmup_factor = self.hparams.learning_rate_warmup_factor
        print("  start_decay_step=%d, learning_rate=%g, decay_steps %d, "
              "decay_factor %g, learning_rate_warmup_steps=%d, "
              "learning_rate_warmup_factor=%g, starting_learning_rate=%g" %
              (self.hparams.start_decay_step, self.hparams.learning_rate, self.hparams.decay_steps,
               self.hparams.decay_factor, warmup_steps, warmup_factor,
               (self.hparams.learning_rate * warmup_factor ** warmup_steps)))
        self.global_step = tf.Variable(0, trainable=False)

        params = tf.trainable_variables()
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.learning_rate = tf.constant(self.hparams.learning_rate)
            #inv_decay = warmup_factor ** (
            #    tf.to_float(warmup_steps - self.global_step))
            #self.learning_rate = tf.cond(
            #    self.global_step < self.hparams.learning_rate_warmup_steps,
            #    lambda: inv_decay * self.learning_rate,
            #    lambda: self.learning_rate,
            #    name="learning_rate_decay_warump_cond")

            if self.hparams.optimizer == "sgd":
                self.learning_rate = tf.cond(
                    self.global_step < self.hparams.start_decay_step,
                    lambda: self.learning_rate,
                    lambda: tf.train.exponential_decay(
                        self.learning_rate,
                        (self.global_step - self.hparams.start_decay_step),
                        self.hparams.decay_steps,
                        self.hparams.decay_factor,
                        staircase=True),
                    name="learning_rate")
                self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
                tf.summary.scalar("lr", self.learning_rate)
            elif self.hparams.optimizer == "adam":
                assert float(
                    self.hparams.learning_rate
                ) <= 0.001, "! High Adam learning rate %g" % self.hparams.learning_rate
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

            gradients = tf.gradients(
                self.train_loss,
                params,
                colocate_gradients_with_ops=self.hparams.colocate_gradients_with_ops)

            clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.hparams.max_gradient_norm)

            self.update = self.optimizer.apply_gradients(
                zip(clipped_gradients, params))#, global_step=self.global_step)



    def train(self):

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("../tmp/deep", self.sess.graph)

        start_time = time.time()
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        self.sess.run(init_g)
        self.sess.run(init_l)
        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for epoch in range(self.hparams.number_of_epochs):
            iteration = 0
            while iteration * self.hparams.batch_size < self.hparams.training_size:
                _, summary_str, cost, accuracy = self.sess.run([self.update, merged, self.train_loss, self.accuracy])

                iteration += 1
                if iteration % 10 == 0:
                    writer.add_summary(summary_str, iteration)
                    print("iterations: [%2d] time: %4.4f, loss: %.8f, accuracy: %.8f" \
                          % (iteration, time.time() - start_time, np.mean(cost), accuracy))


        self.save()
        coord.request_stop()
        coord.join(threads)




if __name__ == '__main__':
    dataset_name = "cnn"
    dataset_dir = "../data"
    dr = DataReader()


    hparams = tf.flags
    hparams.DEFINE_integer("training_size", 80000, "total number of training samples")
    hparams.DEFINE_integer("number_of_epochs", 25, "Epoch to train [25]")
    hparams.DEFINE_integer("vocab_size", 100000, "The size of vocabulary [10000]")
    hparams.DEFINE_integer("batch_size", 32, "The size of batch images [32]")
    hparams.DEFINE_integer("depth", 2, "Depth [1]")
    hparams.DEFINE_integer("max_nsteps", 1000, "Max number of steps [1000]")
    hparams.DEFINE_integer("number_of_hidden_units", 256, "The size of hidden layers")
    hparams.DEFINE_float("learning_rate", 5e-5, "Learning rate [0.00005]")
    hparams.DEFINE_float("momentum", 0.9, "Momentum of RMSProp [0.9]")
    hparams.DEFINE_float("keep_prob", 1.0, "keep_prob [0.5]")
    hparams.DEFINE_float("decay", 0.95, "Decay of RMSProp [0.95]")
    hparams.DEFINE_string("dtype", "float32", "dtype [float32]")
    hparams.DEFINE_string("model", "LSTM", "The type of model to train and test [LSTM, BiLSTM, Attentive, Impatient]")
    hparams.DEFINE_string("data_dir", "../data", "The name of data directory [data]")
    hparams.DEFINE_string("dataset_name", "cnn", "The name of dataset [cnn, dailymail]")
    hparams.DEFINE_string("checkpoint_dir", "checkpoint_bidi", "Directory name to save the checkpoints [checkpoint]")
    hparams.DEFINE_integer("learning_rate_warmup_steps", 100, "How many steps we inverse-decay learning.")
    hparams.DEFINE_float("learning_rate_warmup_factor", 1.0,"The inverse decay factor for each warmup step.")
    hparams.DEFINE_integer("start_decay_step", 10, "When we start to decay")
    hparams.DEFINE_integer("decay_steps",10000, "How frequent we decay")
    hparams.DEFINE_float("decay_factor", 0.98, "How much we decay.")
    hparams.DEFINE_string("optimizer", "adam", "sgd | adam")
    hparams.DEFINE_bool("colocate_gradients_with_ops", True,
                        "Whether try colocating gradients with "
                              "corresponding op")
    hparams.DEFINE_float("max_gradient_norm", 5.0,"Clip gradients to this norm.")
    hparams = hparams.FLAGS

    with tf.Session() as sess:
        deep_lstm_reader = DeepLSTMReader(sess=sess,hparams=hparams, mode=tf.contrib.learn.ModeKeys.TRAIN, data_reader=dr)
        deep_lstm_reader.define_graph()
        deep_lstm_reader._define_train()
        deep_lstm_reader.train()


    #(_, document, question, answer, _), data_idx, data_max_idx = next(data_iterator)

        #deep_lstm_reader.train(sess=sess,data_dir=dataset_dir,dataset_name=dataset_name)
