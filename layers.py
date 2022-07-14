import tensorflow as tf
from tensorflow.keras import layers


NCHW_FORMAT = 'NCHW'
NHWC_FORMAT = 'NHWC'
DEFAULT_DATA_FORMAT = NHWC_FORMAT
"""
From TF 2.9.1 docs:
The dtype of the layer's computations and weights. 
Can also be a tf.keras.mixed_precision.Policy, which allows the computation and weight dtype to differ. 
Default of None means to use tf.keras.mixed_precision.global_policy(), 
which is a float32 policy unless set to different value.
"""
DEFAULT_DTYPE = None

DATA_FORMAT_DICT = {
    NCHW_FORMAT: 'channels_first',
    NHWC_FORMAT: 'channels_last'
}

# For BatchNorm layer axis which should NOT be reduced is provided
BATCH_NORM_AXIS_DICT = {
    DATA_FORMAT_DICT[NCHW_FORMAT]: 1,
    DATA_FORMAT_DICT[NHWC_FORMAT]: 3,
}

to_NCHW_axis = [0, 3, 1, 2] # NHWC -> NCHW
to_NHWC_axis = [0, 2, 3, 1] # NCHW -> NHWC

ACTIVATION_DICT = {
    'relu' : tf.nn.relu,
    'lrelu': tf.nn.leaky_relu,
    'elu'  : tf.nn.elu,
    'selu' : tf.nn.selu,
    'swish': tf.nn.swish,
    'silu' : tf.nn.silu,
    'gelu' : tf.nn.gelu,
}

NORM_DICT = {
    'BN'       : layers.BatchNormalization,
    'BatchNorm': layers.BatchNormalization,
    'LN'       : layers.LayerNormalization,
    'LayerNorm': layers.LayerNormalization
}


def identity():
    return lambda x: x


class FFCSE_block(layers.Layer):

    def __init__(self, channels, ratio_global, data_format=DEFAULT_DATA_FORMAT, dtype=DEFAULT_DTYPE):
        super(FFCSE_block, self).__init__(dtype=dtype)
        in_channels_global = int(channels * ratio_global)
        in_channels_local = channels - in_channels_global
        reduction = 16

        # Output size should (1, 1)
        self.avg_pool = layers.AvgPool2D(pool_size=(channels, channels), strides=(channels, channels),
                                         data_format=DATA_FORMAT_DICT[data_format], dtype=dtype)
        conv_kwargs = {'kernel_size': 1, 'use_bias': True, 'data_format': DATA_FORMAT_DICT[data_format], 'dtype': dtype}
        self.conv_a1 = layers.Conv2D(channels // reduction, **conv_kwargs)
        self.conv_a2_local = None if in_channels_local == 0 else layers.Conv2D(in_channels_local, **conv_kwargs)
        self.conv_a2_global = None if in_channels_global == 0 else layers.Conv2D(in_channels_global, **conv_kwargs)

    def call(self, x, **kwargs):
        x = x if type(x) is tuple else (x, 0)
        id_local, id_global = x

        x = id_local if type(id_global) is int else tf.concat([id_local, id_global], axis=1)
        x = self.avg_pool(x)
        x = tf.nn.relu(self.conv_a1(x))

        x_local = 0 if self.conv_a2_local is None else id_local * tf.nn.sigmoid(self.conv_a2_local(x))
        x_global = 0 if self.conv_a2_global is None else id_global * tf.nn.sigmoid(self.conv_a2_global(x))

        return x_local, x_global


class FourierUnit(layers.Layer):

    def __init__(self, fmaps, groups=1, data_format=DEFAULT_DATA_FORMAT, dtype=DEFAULT_DTYPE):
        super(FourierUnit, self).__init__(dtype=dtype)
        # Note: fmaps is out_channels, in_channels are not needed due to Conv2D layer in Keras
        # Implementation is for NCHW data format, so layers should only be called with NCHW settings
        layers_data_format = DATA_FORMAT_DICT[NCHW_FORMAT]
        self.fmaps = fmaps
        self.conv_block = tf.keras.Sequential([
            layers.Conv2D(fmaps * 2, kernel_size=1, strides=1, padding='SAME', groups=groups, use_bias=False, data_format=layers_data_format, dtype=dtype),
            layers.BatchNormalization(axis=BATCH_NORM_AXIS_DICT[layers_data_format], dtype=dtype),
            layers.Activation(tf.nn.relu, dtype=dtype)
        ], name='FU_conv_block')
        self.data_format = data_format
        self.use_fp16 = self.compute_dtype == 'float16'
        # self.call = tf.function(self.call, jit_compile=True)
        self.print_info = False

    def call(self, x, **kwargs):
        if self.data_format == NHWC_FORMAT:
            x = tf.transpose(x, to_NCHW_axis)

        """
        PyTorch fft and iift:
        # torch.rfft(x, signal_ndim=2, normalized=True)
        # torch.irfft(ffted, signal_ndim=2, signal_sizes=r_size[2:], normalized=True)
        """

        # s = tf.shape(x)
        # N, C, H, W = s[0], s[1], s[2], s[3]
        N = tf.shape(x)[0]
        C, H, W = x.shape[1], x.shape[2], x.shape[3]

        if self.use_fp16:
            x = tf.cast(x, tf.float32)
        # TODO: add normalization. ndim seems to be correct
        norm_const = tf.sqrt(tf.cast(W, tf.float32))
        x /= norm_const
        ffted = tf.signal.rfft(x)                                                # [N, C, H, W/2 + 1] of dtype complex64
        ffted_re, ffted_im = tf.math.real(ffted)[..., tf.newaxis], tf.math.imag(ffted)[..., tf.newaxis]
        ffted = tf.concat([ffted_re, ffted_im], axis=-1)                         # [N, C, H, W/2 + 1, 2]
        ffted = tf.transpose(ffted, perm=[0, 1, 4, 2, 3])                        # [N, C, 2, H, W/2 + 1]
        ffted = tf.reshape(ffted, shape=[-1, C * 2, ffted.shape[-2], ffted.shape[-1]]) # was [N, -1, ...

        ffted = self.conv_block(ffted, **kwargs)                                 # [N, C * 2, H, W/2 + 1]
        if self.use_fp16:
            ffted = tf.cast(ffted, tf.float32) # Prepare values for complex numbers after conv block

        ffted = tf.reshape(ffted, [-1, self.fmaps, 2, ffted.shape[-2], ffted.shape[-1]]) # was [N, -1, 2, ...]
        ffted = tf.transpose(ffted, [0, 1, 3, 4, 2])                             # [N, C, H, W/2 + 1, 2]
        ffted_re, ffted_im = ffted[..., 0], ffted[..., 1]
        ffted = tf.complex(ffted_re, ffted_im)

        # TODO: think about normalization and size of signal
        output = tf.signal.irfft(ffted) * norm_const
        if self.use_fp16:
            output = tf.cast(output, tf.float16)

        if self.data_format == NHWC_FORMAT:
            output = tf.transpose(output, to_NHWC_axis)
        return output


class SpectralTransform(layers.Layer):

    def __init__(self, fmaps, stride=1, groups=1, enable_local_fu=True, data_format=DEFAULT_DATA_FORMAT, dtype=DEFAULT_DTYPE):
        super(SpectralTransform, self).__init__(dtype=dtype)
        # Note: fmaps is out_channels, in_channels are not needed due to Conv2D layer in Keras
        self.stride = stride
        self.enable_local_fu = enable_local_fu
        self.data_format = data_format
        self.use_fp16 = self.compute_dtype == 'float16'

        # Implementation is for NCHW data format, so layers should only be called with NCHW settings
        layers_data_format = DATA_FORMAT_DICT[NCHW_FORMAT]
        if stride == 2:
            self.downsample = layers.AvgPool2D(pool_size=(2, 2), strides=(2, 2), data_format=layers_data_format, dtype=dtype)
        else:
            self.downsample = None

        conv_kwargs = {'groups': groups, 'data_format': layers_data_format, 'dtype': dtype}
        self.conv1 = tf.keras.Sequential([
            layers.Conv2D(fmaps // 2, kernel_size=1, use_bias=False, **conv_kwargs),
            layers.BatchNormalization(axis=BATCH_NORM_AXIS_DICT[layers_data_format], dtype=dtype),
            layers.Activation(tf.nn.relu, dtype=dtype)
        ], name='ST_conv1_block')
        self.fu = FourierUnit(fmaps // 2, **conv_kwargs)
        if self.enable_local_fu:
            self.local_fu = FourierUnit(fmaps // 2, **conv_kwargs)
        self.conv2 = layers.Conv2D(fmaps, kernel_size=1, use_bias=False, **conv_kwargs)

    def call(self, x, **kwargs):
        if self.data_format == NHWC_FORMAT:
            x = tf.transpose(x, to_NCHW_axis)

        if self.downsample is not None:
            x = self.downsample(x)

        x = self.conv1(x, **kwargs)
        output = self.fu(x, **kwargs)

        if self.enable_local_fu:
            C = x.shape[1]
            split_no = 2
            # In TensorFlow number of splits is provided, not size of each chinks like in PyTorch
            xs = tf.concat(tf.split(x[:, : C // 4], split_no, axis=-2), axis=1)
            xs = tf.concat(tf.split(xs, split_no, axis=-1), axis=1)
            xs = self.local_fu(xs, **kwargs)
            xs = tf.tile(xs, [1, 1, split_no, split_no])
        else:
            xs = 0

        output = self.conv2(x + output + xs)
        if self.data_format == NHWC_FORMAT:
            output = tf.transpose(output, to_NHWC_axis)
        return output


class FFC(layers.Layer):

    def __init__(self, fmaps_in, fmaps, kernel_size, ratio_global_in, ratio_global_out,
                 stride=1, padding='SAME', dilation_rate=1, groups=1, use_bias=False,
                 enable_local_fu=True, data_format=DEFAULT_DATA_FORMAT, dtype=DEFAULT_DTYPE):
        super(FFC, self).__init__(dtype=dtype)
        # Note: fmaps is out_channels, fmaps_in is in_channels

        assert stride in [1, 2], "Stride should be 1 or 2."
        self.stride = stride
        self.data_format = data_format

        assert 0 <= ratio_global_in <= 1, "ratio_global_in should be from [0, 1]."
        assert 0 <= ratio_global_out <= 1, "ratio_global_out should be from [0, 1]."
        self.ratio_global_in = ratio_global_in
        self.ratio_global_out = ratio_global_out

        self.fmaps_in_global = int(fmaps_in * ratio_global_in)
        self.fmaps_in_local = fmaps_in - self.fmaps_in_global
        self.fmaps_global = int(fmaps * ratio_global_out)
        self.fmaps_local = fmaps - self.fmaps_global

        self.conv_kwargs = {
            'kernel_size': kernel_size, 'strides': stride, 'padding': padding,
            'dilation_rate': dilation_rate, 'groups': groups, 'use_bias': use_bias,
            'data_format': DATA_FORMAT_DICT[data_format], 'dtype': dtype
        }
        self.conv_g2g_kwargs = {
            'stride': stride,
            'groups': 1 if groups == 1 else groups // 2,
            'enable_local_fu': enable_local_fu,
            'data_format': data_format, 'dtype': dtype
        }

    def build(self, input_shape):
        self.conv_l2l = identity() if self.fmaps_in_local == 0 or self.fmaps_local == 0 else layers.Conv2D(self.fmaps_local, **self.conv_kwargs)
        self.conv_l2g = identity() if self.fmaps_in_local == 0 or self.fmaps_global == 0 else layers.Conv2D(self.fmaps_global, **self.conv_kwargs)
        # If x_global is None then conv_g2l and conv_g2g should never be created,
        # otherwise layers has weights which are never created.
        # To check if input has global part it is enough to see len of input_shape (or check if type is tuple of TensorShape)
        if len(input_shape) == 2:
            # Global part of input exists:
            self.conv_g2l = identity() if self.fmaps_in_global == 0 or self.fmaps_local == 0 else layers.Conv2D(
                self.fmaps_local, **self.conv_kwargs)
            self.conv_g2g = identity() if self.fmaps_in_global == 0 or self.fmaps_global == 0 else SpectralTransform(
                self.fmaps_global, **self.conv_g2g_kwargs)

    def call(self, x, **kwargs):
        x_local, x_global = x if type(x) is tuple else (x, None)
        out_x_local, out_x_global = 0, 0

        if self.ratio_global_out != 1:
            out_x_local = self.conv_l2l(x_local) + (0 if x_global is None else self.conv_g2l(x_global))
        if self.ratio_global_out != 0:
            out_x_global = self.conv_l2g(x_local) + (0 if x_global is None else self.conv_g2g(x_global))

        return out_x_local, out_x_global


class FFC_NORM_ACT(layers.Layer):

    def __init__(self, in_channels, out_channels, kernel_size, ratio_global_in, ratio_global_out,
                 stride=1, padding='SAME', dilation_rate=1, groups=1, use_bias=False,
                 norm_layer='BN', activation=None, enable_local_fu=True,
                 data_format=DEFAULT_DATA_FORMAT, dtype=DEFAULT_DTYPE):
        super(FFC_NORM_ACT, self).__init__(dtype=dtype)

        self.ffc = FFC(in_channels, out_channels, kernel_size=kernel_size,
                       ratio_global_in=ratio_global_in, ratio_global_out=ratio_global_out,
                       stride=stride, padding=padding, dilation_rate=dilation_rate, groups=groups, use_bias=use_bias,
                       enable_local_fu=enable_local_fu, data_format=data_format, dtype=dtype)

        def create_norm_layer():
            layer = NORM_DICT.get(norm_layer.lower(), None)
            if isinstance(layer, layers.BatchNormalization):
                layer = layer(axis=BATCH_NORM_AXIS_DICT[data_format], dtype=dtype)
            elif layer is None:
                layer = identity()
            else:
                layer = layer(dtype=dtype)
            return layer

        self.local_norm = identity() if ratio_global_out == 1 else create_norm_layer()
        self.global_norm = identity() if ratio_global_out == 0 else create_norm_layer()

        if activation is None: activation = identity()
        if isinstance(activation, str): activation = ACTIVATION_DICT[activation.lower()]

        self.local_act = identity() if ratio_global_out == 1 else activation
        self.global_act = identity() if ratio_global_out == 0 else activation

    def call(self, x, **kwargs):
        x_local, x_global = self.ffc(x)
        x_local = self.local_act(self.local_norm(x_local))
        x_global = self.global_act(self.global_norm(x_global))
        return x_local, x_global
