from mlp import MLP
from sklearn.datasets import fetch_mldata
import numpy as np
import argparse
from datetime import datetime
import os

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--learning-rate', type=float, dest='learning_rate', default=.05)
parser.add_argument('-b', '--batch-size', type=int, dest='batch_size', default=128)
parser.add_argument('-s', '--layer-sizes', type=int, dest='layer_sizes', nargs='+', default=[512, 512, 512])
parser.add_argument('-l', '--log-dir', type=str, dest='log_dir', default='trained-model')
parser.add_argument('-n', '--n-epochs', type=int, dest='n_epochs', default=20)
parser.add_argument('--debug', dest='debug', action='store_true')
args = parser.parse_args()
learning_rate = args.learning_rate
batch_size = args.batch_size
layer_sizes = args.layer_sizes
log_dir = os.path.join('/training-logs', args.log_dir)
n_epochs = args.n_epochs
debug = args.debug

mnist = fetch_mldata('MNIST original', data_home='mnist')
X, y = np.int32(mnist.data), np.int32(mnist.target)

model = MLP(n_features=X.shape[1],
            n_classes=np.max(y),
            layer_sizes=layer_sizes)
model.__build_training_graph__(learning_rate=learning_rate,
                               batch_size=batch_size,
                               debug=debug)

print('Training MLP with gradient descent...\n\n')

start_time = datetime.now()
shuffle_data = np.arange(X.shape[0])
for i in range(n_epochs):
    np.random.shuffle(shuffle_data)
    X, y = X[shuffle_data], y[shuffle_data]
    X_train, X_val, y_train, y_val = X[:-500], X[-500:], y[:-500], y[-500:]
    examples_seen = 0
    batches_seen = 0
    training_loss = 0

    print('Beginning training epoch {}'.format(i + 1))

    while batches_seen < np.int32(X_train.shape[0] / batch_size):
        example, target = X_train[examples_seen: examples_seen + batch_size], \
               y_train[examples_seen: examples_seen + batch_size]
        training_loss += model.training_step(example, target)
        examples_seen += batch_size
        batches_seen += 1

    print('\t\t[TIME ELAPSED]: {}'.format(str(datetime.now() - start_time)))
    print('\t\t[AVERAGE LOSS FOR EPOCH]: {}'.format(training_loss / batches_seen))

    val_examples_seen = 0
    val_batches_seen = 0
    val_loss = 0

    while val_batches_seen < np.int32(X_val.shape[0] / batch_size):
        example, target = X_val[val_examples_seen: val_examples_seen + batch_size], \
               y_val[val_examples_seen: val_examples_seen + batch_size]
        val_loss += model.validation_step(example, target)
        val_examples_seen += batch_size
        val_batches_seen += 1

    print('\t\t[CURRENT VALIDATION LOSS]: {}\n'.format(val_loss / val_batches_seen))

print('\nTraining complete.')
