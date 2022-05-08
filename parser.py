import tensorflow as tf
from solution.logistic import softmax, cross_entropy
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main_2():
    with tf.GradientTape() as tape:
        y = tf.random.uniform(shape=(), minval=0, maxval=1, dtype=tf.int32)
        X = tf.Variable(tf.random.normal(shape=(10, 10),
                                         mean=0, stddev=0.01))

        W = tf.Variable(tf.random.normal(shape=(100, 2),
                                         mean=0, stddev=0.01))

        b = tf.Variable(tf.zeros(2))
        flatten_x = tf.reshape(X, (1, -1))
        out = softmax(tf.matmul(flatten_x, W) + b)
        one_hot = tf.one_hot(y, 1)
        loss = cross_entropy(out, one_hot)
        grad = tape.gradient(loss, [W, b])
        print(grad)
#if __name__ == "__main__":
#    main()


def main():
    with tf.GradientTape() as tape:
        x = tf.Variable([2.0, 4.])
        tape.watch(x)
        y = x ** 3
        print(tape.gradient(y, x))

if __name__ == "__main__":
    print(-0.5814829 + 0.01594984, -0.56553304)
    main()


