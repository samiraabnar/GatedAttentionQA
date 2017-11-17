import re
import time
import sys
import os
from glob import glob
from tqdm import *
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from gensim import corpora
import numpy as np
import tensorflow as tf
from utils import Util


class DataReader(object):
    # Regular expressions used to tokenize.
    _WORD_SPLIT = re.compile("([.,!?\"':;)(])")
    _DIGIT_RE = re.compile(r"(^| )\d+")

    _ENTITY = "@entity"
    _BAR = "_BAR"
    _UNK = "_UNK"
    BAR_ID = 0
    UNK_ID = 1
    _START_VOCAB = [_BAR, _UNK]

    tokenizer = RegexpTokenizer(r'@?\w+')
    cachedStopWords = stopwords.words("english")

    def basic_tokenizer(self,sentence):
        """Very basic tokenizer: split the sentence into a list of tokens."""
        words = DataReader.tokenizer.tokenize(sentence)
        return [w for w in words if w not in DataReader.cachedStopWords]

    def create_vocabulary(self,vocabulary_path, context, max_vocabulary_size,
                          tokenizer=None, normalize_digits=True):
        """Create vocabulary file (if it does not exist yet) from data file.
        Data file is assumed to contain one sentence per line. Each sentence is
        tokenized and digits are normalized (if normalize_digits is set).
        Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
        We write it to vocabulary_path in a one-token-per-line format, so that later
        token in the first line gets id=0, second line gets id=1, and so on.
        Args:
          vocabulary_path: path where the vocabulary will be created.
          data_path: data file that will be used to create vocabulary.
          max_vocabulary_size: limit on the size of the created vocabulary.
          tokenizer: a function to use to tokenize each data sentence;
            if None, basic_tokenizer will be used.
          normalize_digits: Boolean; if true, all digits are replaced by 0s.
        """
        if not tf.gfile.Exists(vocabulary_path):
            t0 = time.time()
            print("Creating vocabulary %s" % (vocabulary_path))
            print("max_vocabulary_size: ", max_vocabulary_size)
            texts = [word for word in context.lower().split() if word not in DataReader.cachedStopWords]
            dictionary = corpora.Dictionary([texts], prune_at=max_vocabulary_size)
            dictionary.filter_extremes(no_below=1, no_above=1, keep_n=max_vocabulary_size)
            print("vocab length: ", len(dictionary.token2id))
            print("Tokenize : %.4fs" % (t0 - time.time()))
            dictionary.save(vocabulary_path)

    def initialize_vocabulary(self,vocabulary_path):
        """Initialize vocabulary from file.
        We assume the vocabulary is stored one-item-per-line, so a file:
          dog
          cat
        will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
        also return the reversed-vocabulary ["dog", "cat"].
        Args:
          vocabulary_path: path to the file containing the vocabulary.
        Returns:
          a pair: the vocabulary (a dictionary mapping string to integers), and
          the reversed vocabulary (a list, which reverses the vocabulary mapping).
        Raises:
          ValueError: if the provided vocabulary_path does not exist.
        """
        if tf.gfile.Exists(vocabulary_path):
            vocab = corpora.Dictionary.load(vocabulary_path)
            print("vocab length: ",len(vocab.token2id))

            return vocab.token2id, vocab.token2id.keys()
        else:
            raise ValueError("Vocabulary file %s not found.", vocabulary_path)

    def sentence_to_token_ids(self,sentence, vocabulary,
                              tokenizer=None, normalize_digits=True):
        """Convert a string to list of integers representing token-ids.
        For example, a sentence "I have a dog" may become tokenized into
        ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
        "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].
        Args:
          sentence: a string, the sentence to convert to token-ids.
          vocabulary: a dictionary mapping tokens to integers.
          tokenizer: a function to use to tokenize each sentence;
            if None, basic_tokenizer will be used.
          normalize_digits: Boolean; if true, all digits are replaced by 0s.
        Returns:
          a list of integers, the token-ids for the sentence.
        """
        if tokenizer:
            words = tokenizer(sentence)
        else:
            words = self.basic_tokenizer(sentence)
        if not normalize_digits:
            return [vocabulary.get(w, DataReader.UNK_ID) for w in words]
        # Normalize digits by 0 before looking words up in the vocabulary.
        return [vocabulary.get(re.sub(DataReader._DIGIT_RE, " ", w), DataReader.UNK_ID) for w in words]

    def data_to_token_ids(self,data_path, target_path, vocab,
                          tokenizer=None, normalize_digits=True):
        """Tokenize data file and turn into token-ids using given vocabulary file.
        This function loads data line-by-line from data_path, calls the above
        sentence_to_token_ids, and saves the result to target_path. See comment
        for sentence_to_token_ids on the details of token-ids format.
        Args:
          data_path: path to the data file in one-sentence-per-line format.
          target_path: path where the file with token-ids will be created.
          vocabulary_path: path to the vocabulary file.
          tokenizer: a function to use to tokenize each sentence;
            if None, basic_tokenizer will be used.
          normalize_digits: Boolean; if true, all digits are replaced by 0s.
        """
        # if not gfile.Exists(target_path):
        if True:
            with tf.gfile.GFile(data_path, mode="r") as data_file:
                counter = 0
                results = []
                for line in data_file:
                    if counter == 0:
                        results.append(line)
                    elif counter == 4:
                        entity, ans = line.split(":", 1)
                        try:
                            results.append("%s:%s" % (vocab[entity[:]], ans))
                        except:
                            continue
                    else:
                        token_ids = self.sentence_to_token_ids(line, vocab, tokenizer,
                                                          normalize_digits)
                        results.append(" ".join([str(tok) for tok in token_ids]) + "\n")
                    if line == "\n":
                        counter += 1

                try:
                    len_d, len_q = len(results[2].split()), len(results[4].split())
                except:
                    return
                with open("%s_%s" % (target_path, len_d + len_q), mode="w") as tokens_file:
                    tokens_file.writelines(results)

    def get_all_context(self,dir_name, context_fname):
        context = ""
        for fname in tqdm(glob(os.path.join(dir_name, "*.question"))):
            with open(fname) as f:
                try:
                    lines = f.read().split("\n\n")
                    context += lines[1] + " "
                    context += lines[4].replace(":", " ") + " "
                except:
                    print(" [!] Error occured for %s" % fname)
        print(" [*] Writing %s ..." % context_fname)
        with open(context_fname, 'w') as f:
            f.write(context)
        return context

    def questions_to_token_ids(self,data_path, vocab_fname, vocab_size):
        vocab, _ = self.initialize_vocabulary(vocab_fname)
        for fname in tqdm(glob(os.path.join(data_path, "*.question"))):
            self.data_to_token_ids(fname, fname + ".ids%s" % vocab_size, vocab)

    def prepare_data(self,data_dir, dataset_name, vocab_size):
        train_path = os.path.join(data_dir, dataset_name, 'questions', 'training')

        context_fname = os.path.join(data_dir, dataset_name, '%s.context' % dataset_name)
        vocab_fname = os.path.join(data_dir, dataset_name, '%s.vocab%s' % (dataset_name, vocab_size))

        if not os.path.exists(context_fname):
            print(" [*] Combining all contexts for %s in %s ..." % (dataset_name, train_path))
            context = self.get_all_context(train_path, context_fname)
        else:
            context = tf.gfile.GFile(context_fname, mode="r").read()
            print(" [*] Skip combining all contexts")

        if not os.path.exists(vocab_fname):
            print(" [*] Create vocab from %s to %s ..." % (context_fname, vocab_fname))
            self.create_vocabulary(vocab_fname, context, vocab_size)
        else:
            print(" [*] Skip creating vocab")

        print(" [*] Convert data in %s into vocab indicies..." % (train_path))
        self.questions_to_token_ids(train_path, vocab_fname, vocab_size)



    def load_vocab(self,data_dir, dataset_name, vocab_size):
        vocab_fname = os.path.join(data_dir, dataset_name, "%s.vocab%s" % (dataset_name, vocab_size))
        print(" [*] Loading vocab from %s ..." % vocab_fname)
        return self.initialize_vocabulary(vocab_fname)

    def load_dataset(self,data_dir, dataset_name, vocab_size):
        train_files = glob(os.path.join(data_dir, dataset_name, "questions",
                                        "training", "*.question.ids%s_*" % (vocab_size)))
        max_idx = len(train_files)
        for idx, fname in enumerate(train_files):
            with open(fname) as f:
                yield f.read().split("\n\n"), idx, max_idx


    def get_batch(self,batch_size,vocab_size,max_nsteps,start=False,data_dir=None, dataset_name=None):

        if start == True:
            self.data_iterator = self.load_dataset(data_dir, dataset_name, vocab_size)
        target_outputs = np.zeros([batch_size, vocab_size])
        inputs, nstarts, answers = [], [], []

        data_idx, data_max_idx = 0,0
        for example_id in np.arange(batch_size):
            try:
                (_, document, question, answer, _), data_idx, data_max_idx = next(self.data_iterator)
            except StopIteration:
                break

            data = [int(d) for d in document.split()] + [0] + \
                   [int(q) for q in question.split() for q in question.split()]

            if len(data) > max_nsteps:
                continue

            inputs.append(data)
            nstarts.append(len(inputs[-1]) - 1)
            target_outputs[example_id][int(answer)] = 1

        if(len(inputs) > 0):
            inputs = Util.array_pad(inputs, max_nsteps, pad=0)
            nstarts = [[nstart, idx, 0] for idx, nstart in enumerate(nstarts)]

        return inputs, nstarts, target_outputs, data_idx, data_max_idx


def test1():
    dr = DataReader()

    if len(sys.argv) < 3:
        print(" [*] usage: python data_utils.py DATA_DIR DATASET_NAME VOCAB_SIZE")
    else:
        data_dir = sys.argv[1]
        dataset_name = sys.argv[2]
        if len(sys.argv) > 3:
            vocab_size = sys.argv[3]
        else:
            vocab_size = 100000

            # dr.prepare_data(data_dir, dataset_name, int(vocab_size))
    data = dr.load_dataset("../data", "cnn", 100000)

    print(next(data))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def read_tf_record_file(filename_queue):
    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(filename_queue)

    features = {
        'document': tf.VarLenFeature(tf.int64),
        'question': tf.VarLenFeature(tf.int64),
        'answer': tf.VarLenFeature(tf.int64),
    }

    parsed_example = tf.parse_single_example(serialized_example,features=features)

    return parsed_example['document'],parsed_example['question'],parsed_example['answer'],\
           parsed_example['document'].dense_shape,parsed_example['question'].dense_shape,parsed_example['answer'].dense_shape



def test2(train_files,mode):
    filename_queue = tf.train.string_input_producer(train_files)
    reader = tf.WholeFileReader()
    key, example = reader.read(filename_queue)
    parsed_example = tf.string_split([example], '\n\n')
    filename = os.path.join("../data", "cnn_"+mode+"_0"+ '.tfrecords')
    writer = tf.python_io.TFRecordWriter(filename)

    with tf.Session() as sess:
        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(len(train_files)):
            # Retrieve a single instance:
            if i > 0 and (i % 1000 == 0):
                writer.close()
                filename = os.path.join("../data", "cnn" + "_"+str(i) + '.tfrecords')
                writer = tf.python_io.TFRecordWriter(filename)

            (_, data, _) = sess.run(parsed_example)
            print(data[1])
            document = list(map(int,data[1].decode().split(" "))) + [0]
            question = list(map(int,data[2].decode().split(" "))) + [0]
            answer = list(map(int,data[3].decode().split(" ")))

            print(question)




            feature_list = {
                'document':  _int64_feature(document)
                , 'question': _int64_feature(question)
                , 'answer': _int64_feature(answer)
            }

            feature = tf.train.Features(feature=feature_list)
            example = tf.train.Example(features=feature)



            writer.write(example.SerializeToString())


        writer.close()
        coord.request_stop()
        coord.join(threads)


def parser(record):
    keys_to_features = {
        "document": tf.FixedLenFeature([],tf.string),
        "question": tf.FixedLenFeature([],tf.string),
        "answer": tf.FixedLenFeature([],tf.string)
    }
    parsed = tf.parse_single_example(record, features=keys_to_features)

    # Perform additional preprocessing on the parsed data.
    document = parsed["document"]
    question = parsed["question"]
    answer = parsed["answer"]


    return {"document": document, "question": question}, answer


def reader2():
    filenames = ["../data/cnn_0.tfrecords"]
    batch_size = 10
    min_after_dequeue = 1000

    filename_queue = tf.train.string_input_producer(
        filenames)
    document,question,answer,document_shape,question_shape,answer_shape = read_tf_record_file(filename_queue)

    d_batch, q_batch, ans_batch,document_shape_batch,question_shape_batch,answer_shape_batch= tf.train.shuffle_batch([document,question,answer,document_shape,question_shape,answer_shape], batch_size=batch_size,
                                                     capacity=min_after_dequeue*3+1, min_after_dequeue=min_after_dequeue)
    d_q_batch = tf.sparse_concat(axis=1,sp_inputs=[d_batch,q_batch],)
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
    d_q_lengths = tf.reduce_sum(tf.concat([tf.reshape(document_shape_batch,(batch_size,1)),tf.reshape(question_shape_batch,(batch_size,1))],axis=1),axis=1)





    #d_q_lengths = [ for indice in tf.unstack(d_q_batch.indices)]

    with tf.Session() as sess:
        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(100):
            print(i)
            document_question, answer,lengths = sess.run([dense_d_q_batch,dens_ans_batch,d_q_lengths])
            print(document_question.shape,answer.shape)
            print(lengths)

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':

    vocab_size = 10000
    data_dir = "../data"
    dataset_name="cnn"
    vocab_size=vocab_size
    dr = DataReader()
    # dr.prepare_data(data_dir="../data",
    #                dataset_name="cnn",
    #                vocab_size=vocab_size)


    """
    train_path = os.path.join(data_dir, dataset_name, 'questions', 'validation')
    vocab_fname = os.path.join(data_dir, dataset_name, '%s.vocab%s' % (dataset_name, vocab_size))
    print(" [*] Convert data in %s into vocab indicies..." % (train_path))
    dr.questions_to_token_ids(train_path, vocab_fname, vocab_size)
    """

    #dr = DataReader()
    #dr.prepare_data(data_dir="../data",
    #                dataset_name="cnn",
    #                vocab_size=vocab_size)

    mode = "validation"
    train_files = glob(os.path.join("../data", "cnn", "questions",
                                    mode, "*.question.ids%s_*" % (vocab_size)))
    test2(train_files,mode)

    #reader2()

    #vocab, rev_vocab = dr.load_vocab(data_dir="../data/",
    #                                                         dataset_name="cnn",
    #                                                         vocab_size=10000)
    #vocab_size = len(vocab.keys())
    #print(vocab_size)
