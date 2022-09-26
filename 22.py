from tensorboard.plugins import projector
import os
import tensorflow as tf
import tensorflow_datasets as tfds

# Load the data
(train_data, test_data), info = tfds.load(
    "imdb_reviews/subwords8k",
    split=(tfds.Split.TRAIN, tfds.Split.TEST),
    with_info=True,
    as_supervised=True, )
encoder = info.features["text"].encoder

# Create training batches
train_batches = train_data.shuffle(1000).padded_batch(
    10, padded_shapes=((None,), ())
)

# Create testing batches
test_batches = test_data.shuffle(1000).padded_batch(
    10, padded_shapes=((None,), ())
)

# Get the first batch
train_batch, train_labels = next(iter(train_batches))

# Create an embedding layer
embedding_dim = 16
embedding = tf.keras.layers.Embedding(
    encoder.vocab_size,
    embedding_dim)

# Configure the embedding layer as part of a Keras model
model = tf.keras.Sequential([
    embedding,
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1),])

# Compile the model
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Train the model for a single epoch
history = model.fit(
    train_batches, epochs=1, validation_data=test_batches
)

# Set up a log dir, just like in previous sections
log_dir='/logs/imdb-example/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Save labels separately in a line-by-line manner.
with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
    for subwords in encoder.subwords:
        f.write("{}\n".format(subwords))

# Fill in the rest of the labels with "unknown"
for unknown in range(1, encoder.vocab_size - len(encoder.subwords)):
    f.write("unknown #{}\n".format(unknown))

# Save the weights we want to analyze as a variable
weights = tf.Variable(model.layers[0].get_weights()[0][1:])

# Create a checkpoint from embedding, the filename and key are the
# name of the tensor
checkpoint = tf.train.Checkpoint(embedding=weights)
checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

# Set up config
config = projector.ProjectorConfig()
embedding = config.embeddings.add()

# The name of the tensor will be suffixed by
# `/.ATTRIBUTES/VARIABLE_VALUE`
embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
embedding.metadata_path = 'metadata.tsv'
projector.visualize_embeddings(log_dir, config)

