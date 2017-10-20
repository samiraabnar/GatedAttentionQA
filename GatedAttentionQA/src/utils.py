import tensorflow as tf
import numpy as np

class Util(object):

    @staticmethod
    def gradient_clip(gradients, max_gradient_norm):
      """Clipping gradients of a model."""
      clipped_gradients, gradient_norm = tf.clip_by_global_norm(
          gradients, max_gradient_norm)
      gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm)]
      gradient_norm_summary.append(
          tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients)))

      return clipped_gradients, gradient_norm_summary

    def array_pad(array, width, pad=-1, force=False):
        max_length = max(map(len, array))
        if max_length > width and force != True:
            raise Exception(" [!] Max length of array %s is bigger than given %s" % (max_length, width))
        result = np.full([len(array), width], pad, dtype=np.int64)
        for i, row in enumerate(array):
            for j, val in enumerate(row[:width - 1]):
                result[i][j] = val
        return result