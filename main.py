from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import keras
from keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.utils import load_img, img_to_array, save_img
from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input

from utils import deprocess_image, save_animation

# Load images
content_image_path = Path('data') / 'src' / 'autumn_road.jpg'
style_image_path = Path('data') / 'src' / 'waves.jpg'
output_folder = Path('data') / 'output'

original_width, original_height = keras.utils.load_img(content_image_path).size
print(original_width, original_height)

img_height = 300
img_width = round(original_width * img_height / original_height)
print(img_width, img_height)

base_image = img_to_array(load_img(content_image_path, target_size=(img_height, img_width), interpolation='bicubic'))
style_image = img_to_array(load_img(style_image_path, target_size=(img_height, img_width), interpolation='bicubic'))

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].imshow(base_image / 255.0)
ax[0].axis('off')
ax[0].set_title('Base Image')
ax[1].imshow(style_image / 255.0)
ax[1].axis('off')
ax[1].set_title('Style Image')

fig.show()

# Define VGG model
vgg = VGG19(weights='imagenet', include_top=False)
output_dict = {layer.name: layer.output for layer in vgg.layers}
model = Model(inputs=vgg.input, outputs=output_dict)


def content_loss_fn(base_features, combined_features):
    """Compute MSE loss between `base_features` and `combined_features`"""
    return tf.reduce_sum(tf.square(base_features - combined_features))


def variance_loss_fn(img):
    height, width = img.shape[1:3]
    a = tf.square(img[:, height - 1, :width - 1, :] - img[:, 1:, :width - 1, :])
    b = tf.square(img[:, height - 1, :width - 1, :] - img[:, :height - 1, 1:, :])

    return tf.reduce_sum(a + b)


def style_loss_fn(style_img, combined_img):
    def gram_matrix(feature_matrix):
        flattened_feature_matrix = K.batch_flatten(K.permute_dimensions(feature_matrix, (2, 0, 1)))
        gram = K.dot(flattened_feature_matrix, K.transpose(flattened_feature_matrix))
        return gram

    style_img_gram_matrix = gram_matrix(style_img)
    combined_img_gram_matrix = gram_matrix(combined_img)

    return tf.reduce_sum(tf.square(style_img_gram_matrix - combined_img_gram_matrix))


# Parameters
content_loss_weight = 1e-5
style_loss_weight = 2.5e-1
variance_loss_weight = 1e-5

init_methods = ['content', 'style', 'noise']
reconstruction_types = ['content', 'style', 'both']

init_method = 'content'  # One of 'content', 'style' or 'noise'
reconstruction_type = 'both'  # One of 'content', 'style' or 'both'
epochs = 2000
num_epochs_per_save = 100
learning_rate = 2.0

style_feature_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
content_feature_layer = 'block2_conv2'

base_image_tensor = preprocess_input(np.expand_dims(base_image.copy(), axis=0))
style_image_tensor = preprocess_input(np.expand_dims(style_image.copy(), axis=0))

if init_method not in init_methods:
    raise ValueError('Initialization method {init_method} not supported, choose one of {*init_methods}')

if reconstruction_type not in reconstruction_types:
    raise ValueError('Reconstructing {reconstruction_type} not supported, choose one of {*reconstruction_methods}')

if init_method == 'content':
    combined_image_tensor = tf.Variable(base_image_tensor.copy())
elif init_method == 'noise':
    combined_image_tensor = tf.Variable(np.random.random(base_image_tensor.shape), dtype=np.float32)
else:
    combined_image_tensor = tf.Variable(style_image_tensor.copy())

content_losses = []
style_losses = []
variance_losses = []

total_losses = []

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

for epoch in range(epochs + 1):
    with tf.GradientTape() as tp:
        input_tensor = tf.concat([base_image_tensor, style_image_tensor, combined_image_tensor], 0)
        features = model(input_tensor)

        content_features_base_image = features[content_feature_layer][0]
        content_features_combined_image = features[content_feature_layer][2]

        content_loss = content_loss_weight * content_loss_fn(content_features_base_image, content_features_combined_image)

        style_loss = 0.0
        for layer in style_feature_layers:
            style_features_style_image = features[layer][1]
            style_features_combined_image = features[layer][2]
            channels = 3
            size = img_height * img_width
            style_loss_l = style_loss_fn(style_features_style_image, style_features_combined_image)
            style_loss_l = style_loss_l / (4.0 * (channels ** 2) * (size ** 2))
            style_loss += style_loss_l / (len(style_feature_layers))

        style_loss = style_loss_weight * style_loss

        variance_loss = variance_loss_fn(combined_image_tensor)
        variance_loss = variance_loss_weight * variance_loss

        content_losses.append(content_loss)
        style_losses.append(style_loss)
        variance_losses.append(variance_loss)

        if reconstruction_type == 'content':
            total_loss = content_loss
        elif reconstruction_type == 'style':
            total_loss = style_loss
        else:
            total_loss = content_loss + style_loss + variance_loss

        total_losses.append(total_loss.numpy())

    print(f"[Epochs: {epoch}/{epochs}] Total loss: {total_loss:.2f}, content loss: {content_loss:.2f}, style loss: {style_loss:.4f}, variance loss: {variance_loss:.2f}")
    grad = tp.gradient(total_loss, combined_image_tensor)
    optimizer.apply_gradients([(grad, combined_image_tensor)])

    # Apply Gradients
    if epoch % num_epochs_per_save == 0:
        # Save combined image
        combined_image = combined_image_tensor.numpy()[0]
        combined_image = deprocess_image(combined_image)
        fname = output_folder / f"{content_image_path.stem}_and_{style_image_path.stem}" / reconstruction_type / f'regenerate__{epoch:04}.png'
        fname.parent.mkdir(parents=True, exist_ok=True)
        save_img(fname, combined_image)

save_animation(output_folder, content_image_path, style_image, reconstruction_type)

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax[0, 0].plot(np.log10(total_losses), '-.')
ax[0, 0].set_title('Total loss')
ax[0, 1].plot(np.log10(content_losses), '-.')
ax[0, 1].set_title('Content Loss')
ax[1, 0].plot(np.log10(style_losses), '-.')
ax[1, 0].set_title('Style Loss')
ax[1, 1].plot(np.log10(variance_losses), '-.')
ax[1, 1].set_title('Total variance loss')

fig.title('Training losses')

fig.savefig(output_folder / f"{content_image_path.stem}_and_{style_image_path.stem}" / reconstruction_type / 'training_losses.png')
