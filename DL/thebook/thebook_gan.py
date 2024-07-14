# %% GAN 모델 구현
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# %% 생성자와 판별자 네트워크 생성 
def make_generator_network(n_hidden_units=100, n_output=784, n_input=20):
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(n_input,)),
        tf.keras.layers.Dense(n_hidden_units, activation='leaky_relu', use_bias=False),
        tf.keras.layers.Dense(n_output, activation='tanh')
    ])
    
    return model

def make_discriminator_network(n_hidden_units=100, n_input=784):
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(n_input,)),
        tf.keras.layers.Dense(n_hidden_units, activation='leaky_relu'),
        tf.keras.layers.Dense(1, activation=None)
    ])
    
    return model


# %% 모델 생성 및 확인
image_size = (28, 28)
z_size = 20
mode_z = 'uniform'

gen_hidden_units = 100
disc_hidden_units = 100

generator = make_generator_network(n_hidden_units=gen_hidden_units, n_output=np.prod(image_size), n_input=z_size)
generator.summary()

discriminator = make_discriminator_network(n_hidden_units=disc_hidden_units, n_input=np.prod(image_size))
discriminator.summary()


# %% 훈련 데이터셋 정의
def preprocess(ex, mode_z='uniform'):
    image = ex['image']
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.reshape(image, [-1])
    image = image * 2 - 1.0
    
    if mode_z == 'uniform':
        z = tf.random.uniform([z_size], minval=-1.0, maxval=1.0)
    else:
        z = tf.random.normal([z_size])
    
    return z, image

mnist_bldr = tfds.builder('mnist')
mnist_bldr.download_and_prepare()
mnist = mnist_bldr.as_dataset(shuffle_files=False)

mnist_trainset = mnist['train']
mnist_trainset = mnist_trainset.map(preprocess)

# %%
mnist_trainset = mnist_trainset.batch(32, drop_remainder=True)
input_z, input_real = next(iter(mnist_trainset))
print(input_z.shape, input_real.shape)

g_output = generator(input_z)
print(g_output.shape)

d_logits_real = discriminator(input_real)
d_logits_fake = discriminator(g_output)
print(d_logits_real.shape, d_logits_fake.shape)

# %%
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

## 생성자 손실함수
g_labels_real = tf.ones_like(d_logits_fake)

g_loss = loss_fn(y_true=g_labels_real, y_pred=d_logits_fake)

## 판별자 손실함수
d_labels_real = tf.ones_like(d_logits_real)
d_labels_fake = tf.zeros_like(d_logits_fake)

d_loss_real = loss_fn(y_true=d_labels_real, y_pred=d_logits_real)
d_loss_fake = loss_fn(y_true=d_labels_fake, y_pred=d_logits_fake)

print(g_loss, d_loss_real, d_loss_fake)