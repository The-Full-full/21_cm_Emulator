import copy
import h5py
import numpy as np
import tensorflow as tf
from scipy.interpolate import splrep, splev, RegularGridInterpolator
from tensorflow import keras
from scipy.ndimage import gaussian_filter1d
# LR_PATH = '/Users/hovavlazare/GITs/FDMemu/fusion_model/data/lr.npy'
# LR = np.load(LR_PATH)



class CustomLearningRateScheduler(keras.callbacks.Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.

    Arguments:
        schedule: a function that takes an epoch index
            (integer, indexed from 0) and current learning rate
            as inputs and returns a new learning rate as output (float).
    """

    def __init__(self, schedule):
        super().__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch)
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print("\nEpoch %03d: Learning rate is %1.9f." % (epoch, scheduled_lr))

#
# def py21cmEMU_schedule(epoch):
#     return LR[epoch]


def create_model(input_dim, hidden_dims, FC_layer_size, out_dim, activation, name, use_BatchNorm=False):
    input_layer = tf.keras.Input(shape=(input_dim,))
    next_layer = tf.keras.layers.Dense(FC_layer_size, activation=activation)(input_layer)
    for dim in range(hidden_dims - 1):
        next_layer = tf.keras.layers.Dense(FC_layer_size, activation=activation)(next_layer)
    last_common_layer = tf.keras.layers.Dense(out_dim, activation=activation)(next_layer)

    ps_output = build_cnn(last_common_layer, activation=activation)

    xHI_output = build_FC_nn(last_common_layer, activation=activation, out_dim=92, name='xHI')

    # tau_output = build_FC_nn(last_common_layer, out_dim=1, name='tau', activation=activation)
    #
    # Tb_output = build_FC_nn(last_common_layer, out_dim=92, name='Tb', activation=activation)
    #
    # Tk_output = build_FC_nn(last_common_layer, out_dim=92, name='Tk', activation=activation)
    #
    # Ts_output = build_FC_nn(last_common_layer, FC_layer_size=400, out_dim=92, name='Ts',
    #                         activation=activation)  # , activation='linear', use_custom_act=True)
    #
    # UVLF_common = build_FC_nn(last_common_layer, out_dim=256, name='UVLF_common')
    #
    # UVLF_z4 = build_FC_nn(UVLF_common, out_dim=50, FC_layer_size=100, name='UVLFz4', activation=activation)
    #
    # UVLF_z5 = build_FC_nn(UVLF_common, out_dim=50, FC_layer_size=100, name='UVLFz5', activation=activation)
    #
    # UVLF_z6 = build_FC_nn(UVLF_common, out_dim=50, FC_layer_size=100, name='UVLFz6', activation=activation)
    #
    # UVLF_z7 = build_FC_nn(UVLF_common, out_dim=50, FC_layer_size=100, name='UVLFz7', activation=activation)
    #
    # UVLF_z8 = build_FC_nn(UVLF_common, out_dim=50, FC_layer_size=100, name='UVLFz8', activation=activation)
    #
    # UVLF_z9 = build_FC_nn(UVLF_common, out_dim=50, FC_layer_size=100, name='UVLFz9', activation=activation)
    #
    # UVLF_z10 = build_FC_nn(UVLF_common, out_dim=50, FC_layer_size=100, name='UVLFz10', activation=activation)

    demo_model = tf.keras.Model(input_layer, outputs=[ps_output, xHI_output,
                                                      # tau_output,
                                                      # Tb_output,
                                                      # Tk_output,
                                                      # Ts_output,
                                                      # UVLF_z4, UVLF_z5,
                                                      # UVLF_z6, UVLF_z7,
                                                      # UVLF_z8,
                                                      # UVLF_z9,
                                                      # UVLF_z10
                                                      ]
                                                      , name=name)
    demo_model.summary()
    return demo_model


def interp_LF_at_z(LF, Muv_bins):
    # TODO: save the Muv bins as an emulator members

    Muv_bins_orig = np.linspace(-20, -12.5, 30)
    tck = splrep(Muv_bins_orig, LF)
    LF_interp = splev(Muv_bins, tck)
    return LF_interp


def build_FC_nn(input_layer, out_dim, name, hidden_dims=4, FC_layer_size=256, activation='LeakyReLU',
                use_custom_act=False):
    next_layer = tf.keras.layers.Dense(FC_layer_size, activation=activation)(input_layer)
    for dim in range(hidden_dims - 1):
        next_layer = tf.keras.layers.Dense(FC_layer_size, activation=activation)(next_layer)
        if use_custom_act:
            assert activation == 'linear', 'if you wish to use custom activation please set activation to linear'
            next_layer = CustomLayer(units=FC_layer_size, trainable=True)(next_layer)

    if name != 'UVLF_common':
        output_layer = tf.keras.layers.Dense(out_dim, activation='linear', name=name + '_output')(next_layer)
    else:
        output_layer = tf.keras.layers.Dense(out_dim, activation=activation, name=name)(next_layer)
    return output_layer


def build_cnn(input_layer, activation='LeakyReLU'):
    """
    :param activation:
    :param input_layer: the input layer has to be 1D and the function will transform it to the desired dimension
    :return: output with the correct size (k X z)
    """

    d2_input = tf.keras.layers.Reshape((1, 1, 1024))(input_layer)
    next_layer = tf.keras.layers.Conv2DTranspose(256, (4, 2), activation=activation)(d2_input)
    # next_layer = tf.keras.layers.BatchNormalization()(next_layer)
    next_layer = tf.keras.layers.Conv2DTranspose(256, (7, 3), activation=activation, padding='same')(next_layer)
    # next_layer = tf.keras.layers.BatchNormalization()(next_layer)
    next_layer = tf.keras.layers.Conv2DTranspose(256, (3, 3), activation=activation)(next_layer)
    # next_layer = tf.keras.layers.BatchNormalization()(next_layer)
    next_layer = tf.keras.layers.Conv2DTranspose(128, (7, 3), activation=activation, padding='same')(next_layer)
    # next_layer = tf.keras.layers.BatchNormalization()(next_layer)
    next_layer = tf.keras.layers.Conv2DTranspose(128, (7, 3), activation=activation)(next_layer)
    # next_layer = tf.keras.layers.BatchNormalization()(next_layer)
    next_layer = tf.keras.layers.Conv2DTranspose(64, (3, 1), activation=activation)(next_layer)
    # next_layer = tf.keras.layers.BatchNormalization()(next_layer)
    next_layer = tf.keras.layers.Conv2DTranspose(64, (5, 3), activation=activation, padding='same')(next_layer)
    # next_layer = tf.keras.layers.BatchNormalization()(next_layer)
    next_layer = tf.keras.layers.Conv2DTranspose(32, (7, 3), activation=activation, padding='same')(next_layer)
    # next_layer = tf.keras.layers.BatchNormalization()(next_layer)
    next_layer = tf.keras.layers.Conv2DTranspose(32, (7, 3), activation=activation, padding='same')(next_layer)
    # next_layer = tf.keras.layers.BatchNormalization()(next_layer)
    next_layer = tf.keras.layers.UpSampling2D()(next_layer)
    next_layer = tf.keras.layers.Conv2DTranspose(8, (5, 1), activation=activation)(
        next_layer)  # turn the 5 here into 3 if you want 60X12
    # next_layer = tf.keras.layers.BatchNormalization()(next_layer)
    next_layer = tf.keras.layers.Conv2DTranspose(8, (9, 3), activation=activation, padding='same')(next_layer)
    # next_layer = tf.keras.layers.BatchNormalization()(next_layer)
    next_layer = tf.keras.layers.UpSampling2D(size=(2, 1))(next_layer)

    next_layer = tf.keras.layers.Conv2DTranspose(8, (9, 3), activation=activation, padding='same')(next_layer)
    # next_layer = tf.keras.layers.BatchNormalization()(next_layer)
    output = tf.keras.layers.Conv2DTranspose(1, (11, 3), activation=activation, padding='same', name='ps_output')(
        next_layer)

    # d2_input = tf.reshape(input_layer, (-1, 1, 1, 1024))
    # next_layer = tf.keras.layers.BatchNormalization(trainable=True)(d2_input)
    # next_layer = tf.keras.layers.Conv2DTranspose(512, (2, 2), activation=activation)(next_layer)
    # next_layer = tf.keras.layers.BatchNormalization(trainable=True)(next_layer)
    # next_layer = tf.keras.layers.Conv2DTranspose(256, (3, 3), activation=activation)(next_layer)
    # next_layer = tf.keras.layers.BatchNormalization(trainable=True)(next_layer)
    # next_layer = tf.keras.layers.Conv2DTranspose(128, (3, 3), activation=activation)(next_layer)
    # next_layer = tf.keras.layers.BatchNormalization(trainable=True)(next_layer)
    #
    # next_layer = tf.keras.layers.Conv2DTranspose(64, (2, 2),strides=(2,2), activation=activation)(next_layer)
    # next_layer = tf.keras.layers.BatchNormalization(trainable=True)(next_layer)
    # next_layer = tf.keras.layers.Conv2D(1, (3, 3), padding='same', activation=activation)(next_layer)
    # next_layer = tf.keras.layers.BatchNormalization(trainable=True)(next_layer)
    # next_layer = tf.keras.layers.Conv2DTranspose(32, (3, 3), activation=activation)(next_layer)
    # next_layer = tf.keras.layers.BatchNormalization(trainable=True)(next_layer)
    #
    # next_layer = tf.keras.layers.Conv2DTranspose(16, (2, 2),strides=(2,2), activation=activation)(next_layer)
    # next_layer = tf.keras.layers.BatchNormalization(trainable=True)(next_layer)
    # next_layer = tf.keras.layers.Conv2D(1, (3, 3), padding='same', activation=activation)(next_layer)
    # next_layer = tf.keras.layers.BatchNormalization(trainable=True)(next_layer)
    # next_layer = tf.keras.layers.Conv2DTranspose(8, (3, 3), activation=activation)(next_layer)
    # next_layer = tf.keras.layers.BatchNormalization(trainable=True)(next_layer)
    #
    # next_layer = tf.keras.layers.Conv2DTranspose(4, (2, 2),strides=(2,2), activation=activation)(next_layer)
    # next_layer = tf.keras.layers.BatchNormalization(trainable=True)(next_layer)
    # next_layer = tf.keras.layers.Conv2D(1, (3, 3), padding='same', activation=activation)(next_layer)
    # next_layer = tf.keras.layers.BatchNormalization(trainable=True)(next_layer)
    # next_layer = tf.keras.layers.Conv2DTranspose(2, (3, 3), activation=activation)(next_layer)
    # next_layer = tf.keras.layers.BatchNormalization(trainable=True)(next_layer)
    #
    # next_layer = tf.keras.layers.Conv2DTranspose(1, (3, 3), activation=activation)(next_layer)
    # next_layer = tf.keras.layers.BatchNormalization(trainable=True)(next_layer)
    # output = tf.keras.layers.Conv2D(1, (3, 3), padding='same', activation='linear', name='ps_output')(next_layer)

    return output


class FCemu:
    def __init__(self,
                 z_glob=None,
                 z_PS=None,
                 k_PS=None,
                 param_names=None,
                 input_dim=9,
                 hidden_dims_FC=7,
                 FC_layer_size=1024,
                 activation='LeakyReLU',
                 name="my_model",
                 restore=False,
                 files_dir=''
                 ):

        self.k_PS = k_PS
        self.z_PS = z_PS
        self.learning_rate = None
        if not restore:
            # self.ps_tr_std = ps_std
            # self.ps_tr_mean = ps_mean
            self.param_names = param_names
            self.z_glob = z_glob
            self.activation = activation
            self.name = name
            self.FC_layer_size = FC_layer_size
            self.hidden_dims_FC = hidden_dims_FC
            self.input_dim = input_dim
            self.NN = create_model(input_dim=self.input_dim,
                                   hidden_dims=self.hidden_dims_FC,
                                   FC_layer_size=self.FC_layer_size, out_dim=1024,
                                   activation=self.activation, name=self.name)

            self.tr_params_min = -np.inf
            self.tr_params_max = np.inf
            self.init_learning_rate = 0.001
            self.mean_dict = dict()
            self.std_dict = dict()
            self.output_names = ['ps', 'xHI',
                                 'tau',
                                 'Tb',
                                 'Tk',
                                 'Ts',
                                 'UVLFz4', 'UVLFz5',
                                 'UVLFz6', 'UVLFz7',
                                 'UVLFz8',
                                 'UVLFz9',
                                 'UVLFz10'
                                 ]
        else:
            assert len(files_dir) > 0, 'if you wish to restore a model please supply the model files directory'
            self.restore(files_dir, name)

        # self.mean_dict['ps'] = ps_mean
        # self.std_dict['ps'] = ps_std

    @staticmethod
    def interp_ps(ps_pred, z_arr, k_2d_arr, z_bins, k_bins_orig):
        """
        :param z_arr: redshift to estimate the power spectra at
        :param k_2d_arr: the k bins one wants to obtain at each redshift
        :return: interpolated power spectrum

        TODO: save the z and k bins as an emulator members
        """

        z_orig = z_bins
        # z_orig = self.z_glob
        points = (z_orig, k_bins_orig)

        ps_output = []

        for i, z_new in enumerate(z_arr):
            interp = RegularGridInterpolator(points, ps_pred)
            pts = np.array([[z_new, k_val] for k_val in k_2d_arr[i]])
            ps_output += [interp(pts)]

        return ps_output

    def preprocess_params(self, X_Tr):
        if not np.any(np.isfinite(self.tr_params_max)):
            self.tr_params_max = np.max(X_Tr, axis=0)
            self.tr_params_min = np.min(X_Tr, axis=0)

        X_Tr -= self.tr_params_min
        X_Tr /= (self.tr_params_max - self.tr_params_min)
        X_Tr = X_Tr * 2 - 1
        return X_Tr

    def preprocess_features(self, Y_tr):
        # Y_tr[0] = np.log10(Y_tr[0])
        # for i, feature in enumerate(Y_tr):
        #     Y_tr[i] = np.array(Y_tr[i])
        Y_tr[4] = np.log10(Y_tr[4])
        Y_tr[5] = np.log10(Y_tr[5])
        Y_tr[0] = np.log10(Y_tr[0])
        for i, feature in enumerate(Y_tr):
            # if i != 0:
            if self.output_names[i] not in self.mean_dict:
                self.mean_dict[self.output_names[i]] = np.mean(feature)
                self.std_dict[self.output_names[i]] = np.std(feature)
            Y_tr[i] = (np.array(feature) - self.mean_dict[self.output_names[i]]) / self.std_dict[self.output_names[i]]
        return Y_tr

    def postprocess_features(self, Y_pred):
        for i, feature in enumerate(Y_pred):
            Y_pred[i] = feature * self.std_dict[self.output_names[i]] + self.mean_dict[self.output_names[i]]
        Y_pred[1] = gaussian_filter1d(np.clip(Y_pred[1], 0, 1), sigma=1)
        Y_pred[5] = gaussian_filter1d(np.power(10, Y_pred[5]), sigma=1)
        Y_pred[4] = gaussian_filter1d(np.power(10, Y_pred[4]), sigma=1)
        Y_pred[3] = gaussian_filter1d(Y_pred[3], sigma=1)
        Y_pred[0] = np.power(10, Y_pred[0])
        return Y_pred

    # def preprocess_ps(self, Y_tr):
    #     Y_tr[0] = (Y_tr[0] - self.ps_tr_mean) / self.ps_tr_std
    #     Y_tr[1] = np.array(Y_tr[1])
    #     Y_tr[2] = np.array(Y_tr[2])
    #     return Y_tr

    # def postprocess_ps(self, ps_pred):
    #     ps_pred = ps_pred * self.ps_tr_std + self.ps_tr_mean
    #     return ps_pred

    def ARE_loss(self):
        """
                L2 loss function - mean absolute percentage error
                :return: loss function
                """

        def loss_func(y_true, y_pred):
            return tf.keras.losses.MeanAbsolutePercentageError()(y_true, y_pred) / 100

        return loss_func

    def double_mse_loss(self):
        def loss_func(y_true, y_pred):
            return 2 * tf.keras.losses.MeanSquaredError()(y_true, y_pred)

        return loss_func

    def train_model(self, X_train, Y_train, X_val, Y_val,
                    learning_rate=0.001,
                    retrain=False,
                    stop_patience_value=10,
                    decay_patience_value=5,
                    reduce_lr_factor=0.5,
                    batch_size=256,
                    epochs=350,
                    verbose=False
                    ):
        if not retrain:
            self.learning_rate = learning_rate

        X_train = self.preprocess_params(X_train)
        Y_train = self.preprocess_features(Y_train)
        X_val = self.preprocess_params(X_val)
        Y_val = self.preprocess_features(Y_val)

        early_stopping_cb = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=stop_patience_value, min_delta=1e-10, restore_best_weights=True, verbose=1
        )
        reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=decay_patience_value, factor=reduce_lr_factor,
            verbose=1, min_delta=5e-9, min_lr=1e-7)

        callbacks = [early_stopping_cb, reduce_lr_cb] #, CustomLearningRateScheduler(py21cmEMU_schedule)]

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        print(f'Initial learning rate: {optimizer.lr.numpy()}')
        self.NN.compile(optimizer=optimizer,
                        loss={'ps_output': self.double_mse_loss(), 'xHI_output': 'mse',
                              # 'tau_output': 'mse',
                              # 'Tb_output': 'mse',
                              # 'Tk_output': 'mse',
                              # 'Ts_output': 'mse',
                              # 'UVLFz4_output': 'mse',  'UVLFz5_output': 'mse',
                              # 'UVLFz6_output': 'mse', 'UVLFz7_output': 'mse', 'UVLFz8_output': 'mse',
                              # 'UVLFz9_output': 'mse',
                              # 'UVLFz10_output': 'mse'
                              },
                        metrics={'ps_output': tf.keras.metrics.MeanAbsolutePercentageError(),
                                 'xHI_output': tf.keras.metrics.RootMeanSquaredError(),
                                 # 'tau_output': tf.keras.metrics.MeanAbsoluteError()
                                 })
        history = self.NN.fit(x=X_train,
                              y=Y_train,
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_data=(X_val, Y_val),
                              validation_batch_size=batch_size,
                              callbacks=callbacks,
                              verbose=verbose,
                              )
        lr = self.NN.optimizer.lr
        self.learning_rate = lr.numpy()
        print(f'final learning rate {self.learning_rate}')
        return history.history['loss'], history.history['val_loss']



    def predictions_to_dict(self, predictions):
        """
        Convert model predictions to a dictionary.

        :param predictions: List of model predictions.
        :param output_names: List of output names.
        :return: Dictionary where keys are output names and values are model predictions.
        """
        return {name: pred for name, pred in zip(self.output_names, predictions)}
    def save(self, dir_path):
        self.NN.save(F'{dir_path}/{self.NN.name}.h5')

        h5f = h5py.File(f'{dir_path}/model_data.h5', 'w')
        h5f.create_dataset('tr_params_min', data=self.tr_params_min)
        h5f.create_dataset('tr_params_max', data=self.tr_params_max)
        h5f.create_dataset('learning_rate', data=self.learning_rate)
        mean_list = list(self.mean_dict.values())
        std_list = list(self.std_dict.values())
        h5f.create_dataset('mean_dict', data=mean_list)
        h5f.create_dataset('std_dict', data=std_list)
        h5f.create_dataset('z_glob', data=self.z_glob)
        h5f.create_dataset('z_PS', data=self.z_PS)
        h5f.create_dataset('k_PS', data=self.k_PS)
        h5f.create_dataset('params_names', data=self.param_names)
        # h5f.create_dataset('tr_features_mean', data=self.ps_tr_mean)
        # h5f.create_dataset('tr_features_std', data=self.ps_tr_std)
        # h5f.create_dataset('param_labels', data=self.param_labels)
        # h5f.create_dataset('features_band', data=self.features_band)
        h5f.close()

    def restore(self, dir_path, model_name):
        # custom_object = {'CustomLayer': CustomLayer}
        custom_object = {"loss_func": self.ARE_loss(), 'CustomLayer': CustomLayer}
        self.NN = tf.keras.models.load_model(f'{dir_path}/{model_name}.h5', custom_objects=custom_object)

        # self.NN = tf.keras.models.load_model(f'{dir_path}/{model_name}.h5')
        h5f = h5py.File(f'{dir_path}/model_data.h5', 'r')
        self.tr_params_min = h5f['tr_params_min'][:]
        self.tr_params_max = h5f['tr_params_max'][:]
        self.learning_rate = h5f['learning_rate']
        std_list = h5f['std_dict'][:]
        mean_list = h5f['mean_dict'][:]

        self.output_names = ['ps', 'xHI', 'tau', 'Tb',
                             'Tk',
                             'Ts',
                             # 'UVLFz4', 'UVLFz5',
                             'UVLFz6', 'UVLFz7',
                             'UVLFz8',
                             # 'UVLFz9',
                             'UVLFz10']
        self.std_dict = {name: value for name, value in zip(self.output_names, std_list)}
        self.mean_dict = {name: value for name, value in zip(self.output_names, mean_list)}

        self.z_glob = h5f['z_glob'][:]
        # self.z_PS = h5f['z_PS'][:]
        # self.k_PS = h5f['k_PS'][:]

        self.param_names = h5f['params_names'][:]
        self.name = model_name

        # self.ps_tr_mean = h5f['tr_features_mean']
        # self.ps_tr_std = h5f['tr_features_std']
        # self.param_labels = h5py.Dataset.asstr(h5f['param_labels'][:])[:]
        # self.features_band = h5f['features_band'][:]

    def predict(self, test_params):
        params = self.preprocess_params(copy.deepcopy(test_params))
        pred_features = self.NN.predict(params, verbose=False)
        pred_features = self.postprocess_features(pred_features)
        return pred_features


# create_model(9, 3, 512, 1024, 'relu', 'demo_model')

# input = tf.keras.Input(shape=(1,1,512), name="img")
# next_layer = tf.keras.layers.Conv2DTranspose(256, 2, activation="relu")(input)
# output = tf.keras.layers.Conv2DTranspose(128, (2,3), activation="relu")(next_layer)
#
# demo_model = tf.keras.Model(input, output, name="my_model")
# demo_model.summary()


class CustomLayer(tf.keras.layers.Layer):

    def __init__(self, alpha=None,
                 units=512,
                 trainable=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.trainable = trainable
        self.units = units
        self.alpha = alpha

    def build(self, input_shape):
        alpha_init = tf.keras.initializers.GlorotNormal()
        self.alpha = tf.Variable(
            dtype=tf.float32,
            initial_value=alpha_init(shape=(self.units,)),
            trainable=self.trainable,
            name="alpha")
        super().build(input_shape)

    def call(self, inputs):
        elem1 = tf.subtract(1.0, self.alpha)
        elem2 = tf.keras.activations.sigmoid(inputs)
        ptrs = tf.add(self.alpha, tf.math.multiply(elem1, elem2))
        return tf.math.multiply(inputs, ptrs)

    def get_config(self):
        config = super(CustomLayer, self).get_config()
        config.update({
            "alpha": self.get_weights()[0] if self.trainable else self.alpha,
            "trainable": self.trainable,
            'units': self.units
        })
        return config
