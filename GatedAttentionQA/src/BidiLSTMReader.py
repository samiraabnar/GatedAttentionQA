from BaseReaderModel import BaseReaderModel
from DeepLSTMReader import DeepLSTMReader
from DataReader import *
from utils import Util
import cells
import tensorflow as tf
import numpy as np
import time



class BidiLSTMReader(DeepLSTMReader):
    def __init__(self,sess,hparams,mode,data_reader):
        super(BidiLSTMReader, self).__init__(sess,hparams,mode,data_reader)

    def define_graph(self):


        filenames = ["../data/cnn.tfrecords"]
        min_after_dequeue = 1000

        filename_queue = tf.train.string_input_producer(
            filenames)
        document, question, answer, document_shape, question_shape, answer_shape = read_tf_record_file(filename_queue)

        d_batch, q_batch, ans_batch, document_shape_batch, question_shape_batch, answer_shape_batch = tf.train.shuffle_batch(
            [document, question, answer, document_shape, question_shape, answer_shape], batch_size=self.hparams.batch_size,
            capacity=min_after_dequeue * 3 + 1, min_after_dequeue=min_after_dequeue)
        #d_q_batch = tf.sparse_concat(axis=1, sp_inputs=[d_batch, q_batch], )
        dense_d_batch = tf.sparse_to_dense(sparse_indices=d_batch.indices,
                                             output_shape=d_batch.dense_shape,
                                             sparse_values=d_batch.values,
                                             default_value=0,
                                             validate_indices=True,
                                             name=None)

        dense_q_batch = tf.sparse_to_dense(sparse_indices=q_batch.indices,
                                           output_shape=q_batch.dense_shape,
                                           sparse_values=q_batch.values,
                                           default_value=0,
                                           validate_indices=True,
                                           name=None)
        dens_ans_batch = tf.sparse_to_dense(sparse_indices=ans_batch.indices,
                                            output_shape=ans_batch.dense_shape,
                                            sparse_values=ans_batch.values,
                                            default_value=0,
                                            validate_indices=True,
                                            name=None)
        d_lengths = tf.reshape(document_shape_batch, [self.hparams.batch_size])
        q_lengths = tf.reshape(question_shape_batch, [self.hparams.batch_size])
        #d_q_lengths = tf.reduce_sum(tf.concat(
        #    [tf.reshape(document_shape_batch, (self.hparams.batch_size, 1)), tf.reshape(question_shape_batch, (self.hparams.batch_size, 1))],
        #    axis=1), axis=1)


        initializer = tf.contrib.keras.initializers.Orthogonal(gain=1.0,dtype=tf.float32)

        self.embedding = tf.get_variable("embedding",[self.vocab_size,self.hparams.number_of_hidden_units],initializer=initializer,dtype=tf.float32)
        tf.logging.info(self.embedding.get_shape())
        self.docs = dense_d_batch
        self.qs = dense_q_batch
        self.y = dens_ans_batch[:,0]#tf.placeholder(tf.float32, [self.hparams.self.hparams.batch_size, self.vocab_size])

        #unstacked_inputs = tf.unstack(self.inputs,axis=1)
        embedded_docs = [tf.nn.embedding_lookup(self.embedding, self.docs[i] ) for i in range(self.hparams.batch_size)]
        embedded_qs = [tf.nn.embedding_lookup(self.embedding, self.qs[i]) for i in range(self.hparams.batch_size)]

        #embedded_inputs = tf.stack(embedded_inputs)
        tf.summary.histogram("embeddings",self.embedding)

        self.__build_deep_lstm_cell(initializer=initializer)
        d_states_series, d_current_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.stacked_cell,
                                                                             cell_bw=self.stacked_cell,
                                                inputs=tf.stack(embedded_docs),
                                                sequence_length=d_lengths,
                                                initial_state_fw=None,
                                                initial_state_bw=None,
                                                dtype=tf.float32,
                                                parallel_iterations=None,
                                                swap_memory=False,
                                                time_major=False,
                                                scope=None)

        q_states_series, q_current_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.stacked_cell,
                                                                           cell_bw=self.stacked_cell,
                                                                           inputs=tf.stack(embedded_qs),
                                                                           sequence_length=q_lengths,
                                                                           initial_state_fw=None,
                                                                           initial_state_bw=None,
                                                                           dtype=tf.float32,
                                                                           parallel_iterations=None,
                                                                           swap_memory=False,
                                                                           time_major=False,
                                                                           scope=None)

        q_current_state_fw, q_current_state_bw = q_current_state
        q_rep = tf.concat([q_current_state_fw[0][1], q_current_state_bw[0][1]],axis=1)# tf.stack(states)
        self.output_size = self.hparams.depth * self.hparams.number_of_hidden_units * 2
        self.q_rep = tf.reshape(q_rep, [self.hparams.batch_size, self.output_size])


        self.W_att_d = tf.get_variable("W_att_d", [self.output_size,self.output_size,],initializer=initializer,dtype="float32")
        self.W_att_q = tf.get_variable("W_att_q", [self.output_size,self.output_size,],initializer=initializer,dtype="float32")
        self.W_att = tf.get_variable("W_att", [self.output_size,1,],initializer=initializer,dtype="float32")
        #self.B_att = tf.get_variable("B_att", [self.hparams.number_of_hidden_units],initializer=initializer,dtype="float32")

        tf.logging.info(d_states_series)
        d_states_series_fw, d_states_series_bw = d_states_series
        self.sequence_output = tf.unstack(tf.concat([d_states_series_fw,d_states_series_bw],axis=2))
        tf.logging.info(self.sequence_output)
        self.attention_input = [tf.tanh(tf.add(tf.matmul(sequence_output,self.W_att_d),tf.matmul(tf.reshape(q_rep,(1,self.output_size)),self.W_att_q) ))
                                                            for q_rep,sequence_output in zip(tf.unstack(self.q_rep),self.sequence_output)]
        tf.logging.info(self.attention_input)
        self.attention_factors = [tf.softmax(tf.matmul(attention_input, self.W_att)) for attention_input in self.attention_input]
        self.attended_document_states = tf.multiply(self.attention_factors,self.sequence_output)
        tf.logging.info(self.attention_factors)
        tf.logging.info(self.attended_document_states)
        self.document_rep = tf.reduce_mean(self.attended_document_states,axis=1)
        tf.logging.info(self.document_rep)

        self.W_d = tf.get_variable("W_d", [self.output_size,self.vocab_size,],initializer=initializer,dtype="float32")
        self.W_q = tf.get_variable("W_q", [self.output_size,self.vocab_size,],initializer=initializer,dtype="float32")

        tf.summary.histogram("weights", self.W_d)

        self.y_ = tf.tanh(tf.add(tf.matmul(self.document_rep,self.W_d),tf.matmul(self.q_rep,self.W_q)))

        self.train_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.y_,labels=self.y)
        tf.summary.scalar("loss", tf.reduce_mean(self.train_loss))

        correct_prediction = tf.equal(self.y, tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        tf.summary.scalar("accuracy", self.accuracy)


    def __build_deep_lstm_cell(self,initializer):
        self.cell = tf.contrib.rnn.LSTMCell(self.hparams.number_of_hidden_units, forget_bias=0.0, use_peepholes=True, state_is_tuple=True)

        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.cell = tf.nn.rnn_cell.DropoutWrapper(self.cell, output_keep_prob=self.hparams.keep_prob)
        #elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
        #else: #if self.mode == tf.contrib.learn.ModeKeys.INFER:
        self.stacked_cell = tf.contrib.rnn.MultiRNNCell([self.cell] * self.hparams.depth,state_is_tuple=True)

        self.initial_state = self.stacked_cell.zero_state(self.hparams.batch_size, tf.float32)




    def train(self):

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("../tmp/bidi", self.sess.graph)

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
    hparams.DEFINE_integer("training_size", 1000, "total number of training samples")
    hparams.DEFINE_integer("number_of_epochs", 25, "Epoch to train [25]")
    hparams.DEFINE_integer("vocab_size", 100000, "The size of vocabulary [10000]")
    hparams.DEFINE_integer("batch_size", 32, "The size of batch images [32]")
    hparams.DEFINE_integer("depth", 1, "Depth [1]")
    hparams.DEFINE_integer("max_nsteps", 1000, "Max number of steps [1000]")
    hparams.DEFINE_integer("number_of_hidden_units", 256, "The size of hidden layers")
    hparams.DEFINE_float("learning_rate", 5e-5, "Learning rate [0.00005]")
    hparams.DEFINE_float("momentum", 0.9, "Momentum of RMSProp [0.9]")
    hparams.DEFINE_float("keep_prob", 0.8, "keep_prob [0.5]")
    hparams.DEFINE_float("decay", 0.95, "Decay of RMSProp [0.95]")
    hparams.DEFINE_string("dtype", "float32", "dtype [float32]")
    hparams.DEFINE_string("model", "LSTM", "The type of model to train and test [LSTM, BiLSTM, Attentive, Impatient]")
    hparams.DEFINE_string("data_dir", "../data", "The name of data directory [data]")
    hparams.DEFINE_string("dataset_name", "cnn", "The name of dataset [cnn, dailymail]")
    hparams.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
    hparams.DEFINE_integer("learning_rate_warmup_steps", 0, "How many steps we inverse-decay learning.")
    hparams.DEFINE_float("learning_rate_warmup_factor", 1.0,"The inverse decay factor for each warmup step.")
    hparams.DEFINE_integer("start_decay_step", 10, "When we start to decay")
    hparams.DEFINE_integer("decay_steps",10000, "How frequent we decay")
    hparams.DEFINE_float("decay_factor", 0.98, "How much we decay.")
    hparams.DEFINE_string("optimizer", "adam", "sgd | adam")
    hparams.DEFINE_bool("colocate_gradients_with_ops", True,
                        "Whether try colocating gradients with "
                              "corresponding op")
    hparams.DEFINE_float("--max_gradient_norm", 5.0,"Clip gradients to this norm.")
    hparams = hparams.FLAGS

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Session() as sess:
        bidi_lstm_reader = BidiLSTMReader(sess=sess,hparams=hparams, mode=tf.contrib.learn.ModeKeys.TRAIN, data_reader=dr)
        bidi_lstm_reader.define_graph()
        bidi_lstm_reader._define_train()
        bidi_lstm_reader.train()


    #(_, document, question, answer, _), data_idx, data_max_idx = next(data_iterator)

        #deep_lstm_reader.train(sess=sess,data_dir=dataset_dir,dataset_name=dataset_name)
