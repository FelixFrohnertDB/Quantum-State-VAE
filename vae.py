import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import qutip as qt


def train_generator(data, b_s):
    indices = np.arange(len(data))
    batch = []
    while True:
        np.random.shuffle(indices)
        for i in indices:
            batch.append(i)
            if len(batch) == b_s:
                yield data[batch]
                batch = []


def test_generator(data, b_s):
    indices = np.arange(len(data))
    batch = []
    while True:
        np.random.shuffle(indices)
        for i in indices:
            batch.append(i)
            if len(batch) == b_s:
                yield data[batch], data[batch]
                batch = []


class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a p-vector."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, beta_factor, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta_factor
        self.beta_scale = 1
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss")
        self.val_loss_tracker = tf.keras.metrics.Mean(
            name="val_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.val_loss_tracker
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(tf.keras.losses.mean_squared_error(data, reconstruction), axis=-1))

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            total_loss = reconstruction_loss + self.beta_scale * self.beta * kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()
        }

    def test_step(self, data):
        _, _, z_test = self.encoder(data[0])
        test_reconstruction = self.decoder(z_test)
        val_loss = tf.reduce_mean(
            tf.reduce_sum(tf.keras.losses.mean_squared_error(data[0], test_reconstruction), axis=-1))
        self.val_loss_tracker.update_state(val_loss)
        return {"val_loss": self.val_loss_tracker.result()}


def vae_mlp_4x4(latent_dim, act_func, f_act):
    encoder_inputs = tf.keras.Input(shape=16)
    x = encoder_inputs
    x = tf.keras.layers.Dense(16, activation=act_func)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(8, activation=act_func)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(4, activation=act_func)(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    latent_inputs = tf.keras.Input(shape=(latent_dim,))
    x = latent_inputs
    x = tf.keras.layers.Dense(4, activation=act_func)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(8, activation=act_func)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(16, activation=act_func)(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    decoder_outputs = tf.keras.layers.Dense(16, activation=f_act)(x)
    decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")

    return encoder, decoder


def plot_scatter(m_0, m_1, labels, title_x, title_y, title_cbar, alpha):
    fig, ax = plt.subplots()
    im = ax.scatter(m_0, m_1, c=labels, alpha=alpha)
    ax.set_xlabel(title_x)
    ax.set_ylabel(title_y)
    cbar = fig.colorbar(im)
    cbar.set_label(title_cbar)
    plt.show()


def plot_hist(history_dict):
    fig, ax = plt.subplots()
    ax.plot(history_dict["val_val_loss"], ".", label="Validation loss", )
    ax.plot(history_dict["reconstruction_loss"], ".", label="Reconstruction loss")
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss', color='black')
    plt.legend()
    plt.show()


def linear_fit(x, y):
    fit_obj = np.polyfit(x, y, 1)
    fit_pred = np.poly1d(fit_obj)

    return np.reshape(fit_pred(x), (21, 10))


def gen_plot_data(rho_array, vae_trained, scan_arr):
    idx = np.round(np.linspace(0, len(scan_arr) - 1, 21)).astype(int)
    selected_rho = np.zeros((21, 10, 16))
    selected_alpha = np.array([[scan_arr[i]] * 10 for i in idx]).flatten()
    cnt = 0
    for i in idx * 1000:
        selected_rho[cnt] = rho_array[i:i + 10]
        cnt += 1

    selected_rho_reshape = np.reshape(selected_rho, (210, 16))
    z_mean, _, _ = vae_trained.encoder.predict(selected_rho_reshape)
    concur_arr = np.array(
        [qt.concurrence(qt.Qobj(dm.reshape(4, 4), dims=[[2, 2], [2, 2]])) for dm in selected_rho_reshape])

    pred_conc = linear_fit(np.abs(z_mean[:, 0]), concur_arr)
    pred_alpha = linear_fit(z_mean[:, 0], selected_alpha)

    return np.reshape(selected_alpha, (21, 10)), np.reshape(concur_arr, (21, 10)), np.reshape(z_mean,
                                                                                              (21, 10)), np.reshape(
        pred_conc, (21, 10)), np.reshape(pred_alpha, (21, 10))
