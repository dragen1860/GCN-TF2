import  time
import  tensorflow as tf
from    tensorflow.keras import optimizers

from    utils import *
from    models import GCN, MLP
from    config import args

import  os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print('tf version:', tf.__version__)

# Set random seed
seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)



# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(args.dataset)
print('adj:', adj.shape)
print('features:', features.shape)
print('y:', y_train.shape, y_val.shape, y_test.shape)
print('mask:', train_mask.shape, val_mask.shape, test_mask.shape)



# Some preprocessing
# D^-1@X
features = preprocess_features(features) # [49216, 2], [49216], [2708, 1433]
print('features coordinates::', features[0].shape)
print('features data::', features[1].shape)
print('features shape::', features[2])

if args.model == 'gcn':
    # D^-0.5 A D^-0.5
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif args.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, args.max_degree)
    num_supports = 1 + args.max_degree
    model_func = GCN
elif args.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(args.model))

# # Define placeholders
# placeholders = {
#     'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
#     'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
#     'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
#     'labels_mask': tf.placeholder(tf.int32),
#     'dropout': tf.placeholder_with_default(0., shape=()),
#     'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
# }

# Create model
model = GCN(input_dim=features[2][1], output_dim=y_train.shape[1],
                num_features_nonzero=features[1].shape) # [1433]



# # Define model evaluation function
# def evaluate(features, support, labels, mask, placeholders):
#     t_test = time.time()
#     feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
#     outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
#     return outs_val[0], outs_val[1], (time.time() - t_test)
#


label = tf.convert_to_tensor(y_train)
labels_mask = tf.convert_to_tensor(train_mask)
features = tf.SparseTensor(*features)
support = [tf.cast(tf.SparseTensor(*support[0]), dtype=tf.float32)]
num_features_nonzero = features.values.shape
dropout = args.dropout
print(num_features_nonzero, support[0].dtype)


optimizer = optimizers.Adam(lr=1e-3)

cost_val = []

# Train model
for epoch in range(args.epochs):

    t = time.time()

    with tf.GradientTape() as tape:

        # Training step
        loss, acc = model((features, label, labels_mask,support))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


    print(epoch, float(loss))

    # # Validation
    # cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
    # cost_val.append(cost)
    #
    # # Print results
    # print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
    #       "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
    #       "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

#     if epoch > args.early_stopping and cost_val[-1] > np.mean(cost_val[-(args.early_stopping+1):-1]):
#         print("Early stopping...")
#         break
#
# print("Optimization Finished!")
#
# # Testing
# test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
# print("Test set results:", "cost=", "{:.5f}".format(test_cost),
#       "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
