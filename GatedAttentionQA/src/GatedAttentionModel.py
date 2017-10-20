import tensorflow as tf;


class GatedAttentionModel(object):
    def __init__(self,hidden_units):
        self.hidden_units = hidden_units


    def define_graph(self):
        # GRU_Q = self._build_encoder()
        # GRU_D = self._build_encoder()




if __name__ == '__main__':

    dataset = tf.contrib.data.TextLineDataset('~/Downloads/cnn.tar.gz')
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    session = tf.Session()
    session.run(next_element)