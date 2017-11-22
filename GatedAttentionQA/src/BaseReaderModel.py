import os
import tensorflow as tf

class BaseReaderModel(object):

    def __init__(self):
        self.vocab = None
        self.data = None

    def save(self,global_step=None):
        self.saver = tf.train.Saver()

        print(" [*] Saving checkpoints...")
        model_name = type(self).__name__ or "Reader"
        if self.hparams.batch_size:
            model_dir = "%s_%s_%s" % (model_name, self.hparams.dataset_name, self.hparams.batch_size)
        else:
            model_dir = self.hparams.dataset_name

        checkpoint_dir = os.path.join(self.hparams.checkpoint_dir, model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print(checkpoint_dir)
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name), global_step=global_step)

    def load(self):
        model_name = type(self).__name__ or "Reader"
        self.saver = tf.train.Saver()

        print(" [*] Loading checkpoints...")
        if self.hparams.batch_size:
            model_dir = "%s_%s_%s" % (model_name, self.hparams.dataset_name, self.hparams.batch_size)
        else:
            model_dir = self.hparams.dataset_name
        checkpoint_dir = os.path.join(self.hparams.checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
