import platform
import tensorflow as tf
from layers import FFCSE_block, FourierUnit, SpectralTransform, FFC, FFC_NORM_ACT, \
    to_NCHW_axis, to_NHWC_axis, NCHW_FORMAT, NHWC_FORMAT, DEFAULT_DTYPE


def prepare_gpu(mode='auto', memory_limit=None):
    OS_LINUX = 'Linux'
    OS_WIN = 'Windows'

    os_name = platform.system()
    os_message = f'\nScript is running on {os_name}, '

    # Note: change this number based on your GPU
    if memory_limit is None:
        memory_limit = 7600 # for real use. Larger values crash the app when starting (system after reboot, memory usage around 300 Mb)
        # memory_limit = 6000
    set_memory_growth = False
    set_memory_limit = False

    assert mode in ['auto', 'growth', 'limit']
    if mode == 'auto':
        if os_name == OS_LINUX:
            print(os_message + 'all memory is available for TensorFlow')
            return
            # print(os_message + 'memory growth option is used')
            # set_memory_growth = True
        elif os_name == OS_WIN:
            print(os_message + 'memory limit option is used')
            set_memory_limit = True
        else:
            print(
                os_message + f'GPU can only be configured for {OS_LINUX}|{OS_WIN}, '
                f'memory growth option is used'
            )
            set_memory_growth = True
    else:
        if mode == 'growth':
            set_memory_growth = True
        else:
            set_memory_limit = True

    physical_gpus = tf.config.experimental.list_physical_devices('GPU')

    if set_memory_limit:
        if len(physical_gpus) > 0:
            try:
                for gpu in physical_gpus:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [
                            tf.config.experimental.VirtualDeviceConfiguration(
                                memory_limit=memory_limit
                            )
                        ]
                    )
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(
                    f'Physical GPUs: {len(physical_gpus)}, logical GPUs: {len(logical_gpus)}'
                )
                print(f'Set memory limit to {memory_limit} Mbs\n')
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)
        else:
            print('GPU is not available\n')

    if set_memory_growth:
        if len(physical_gpus) > 0:
            for gpu in physical_gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f'Physical GPUs: {len(physical_gpus)} \nSet memory growth\n')
        else:
            print('GPU is not available\n')


def test_spectral_transform():
    N, C, H, W = 8, 32, 128, 128
    x_nchw = tf.random.normal(shape=[N, C, H, W], dtype=tf.float32)

    out_C = C

    st_unit = SpectralTransform(C, out_C, dtype=tf.keras.mixed_precision.Policy('mixed_float16'))
    result = st_unit(x_nchw)
    print('result after SpectralTransform:', result.shape, result.dtype)
    print('--- Spectral transform layer works! ---')


def test_main_layer():
    N, C, H, W = 8, 32, 128, 128
    x_nchw_local = tf.random.normal(shape=[N, C, H, W], dtype=tf.float32)
    x_nchw_global = x_nchw_local + 0.1
    out_C = C

    kernel_size = 3
    ratio_global_in = 0.5
    ratio_global_out = 0.5

    main_layer_nchw = FFC_NORM_ACT(C, out_C,
                                   kernel_size=kernel_size,
                                   ratio_global_in=ratio_global_in,
                                   ratio_global_out=ratio_global_out,
                                   data_format=NCHW_FORMAT)

    result_local_nchw, result_global_nchw = main_layer_nchw((x_nchw_local, x_nchw_global))

    print('--- FFC NORM ACT layer works! ---')


def test_data_formats():
    N, C, H, W = 8, 32, 128, 128
    x_nchw_local = tf.random.normal(shape=[N, C, H, W], dtype=tf.float32)
    x_nhwc_local = tf.transpose(x_nchw_local, to_NHWC_axis)

    x_nchw_global = x_nchw_local + 0.1
    x_nhwc_global = x_nhwc_local + 0.1

    out_C = C
    kernel_size = 3
    ratio_global_in = 0.5
    ratio_global_out = 0.5

    main_layer_nchw = FFC_NORM_ACT(C, out_C,
                                   kernel_size=kernel_size,
                                   ratio_global_in=ratio_global_in,
                                   ratio_global_out=ratio_global_out,
                                   data_format=NCHW_FORMAT)
    main_layer_nchw((x_nchw_local, x_nchw_global), training=False)
    main_layer_nchw_weights = main_layer_nchw.get_weights()
    result_local_nchw, result_global_nchw = main_layer_nchw((x_nchw_local, x_nchw_global))

    main_layer_nhwc = FFC_NORM_ACT(C, out_C,
                                   kernel_size=kernel_size,
                                   ratio_global_in=ratio_global_in,
                                   ratio_global_out=ratio_global_out,
                                   data_format=NHWC_FORMAT)
    main_layer_nhwc((x_nhwc_local, x_nhwc_global), training=False)
    main_layer_nhwc.set_weights(main_layer_nchw_weights)
    result_local_nhwc, result_global_nhwc = main_layer_nhwc((x_nhwc_local, x_nhwc_global))

    result_local_diff = tf.reduce_sum(
        tf.abs(result_local_nchw - tf.transpose(result_local_nhwc, to_NCHW_axis)),
        axis=[0, 1, 2, 3]
    )
    result_global_diff = tf.reduce_sum(
        tf.abs(result_global_nchw - tf.transpose(result_global_nhwc, to_NCHW_axis)),
        axis=[0, 1, 2, 3]
    )
    print('Local diff:', result_local_diff)
    print('Global diff', result_global_diff)
    print('--- NCHW and NHWC data formats work! ---')


def test_grad():
    N, C, H, W = 8, 32, 128, 128
    x_nchw_local = tf.random.normal(shape=[N, C, H, W], dtype=tf.float32)
    x_nchw_global = x_nchw_local + 0.1
    out_C = C

    kernel_size = 3
    ratio_global_in = 0.5
    ratio_global_out = 0.5

    main_layer_nchw = FFC_NORM_ACT(C, out_C,
                                   kernel_size=kernel_size,
                                   ratio_global_in=ratio_global_in,
                                   ratio_global_out=ratio_global_out,
                                   data_format=NCHW_FORMAT)

    with tf.GradientTape() as tape:
        result_local_nchw, result_global_nchw = main_layer_nchw((x_nchw_local, x_nchw_global))
        res = result_local_nchw + result_global_nchw
    grads = tape.gradient(res, main_layer_nchw.trainable_variables)

    print('--- Gradient is computed! ---')


def test_fp16_speed():
    kernel_size = 3
    ratio_global_in = 0.5
    ratio_global_out = 0.5

    N, C, H, W = 8, 3, 224, 224
    if tf.keras.backend.image_data_format() == 'channels_first':
        input_shape = [C, H, W]
    else:
        input_shape = [H, W, C]
    x_inputs = tf.random.uniform(shape=[N] + input_shape, minval=-1, maxval=1, dtype=tf.float32)
    fmaps = 256

    def apply_output_block(x):
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation='sigmoid', dtype=tf.float32)
        ])
        return model(x)

    def apply_ffc_block(x, ch_in, ch_out, dtype=DEFAULT_DTYPE):
        layer = FFC_NORM_ACT(ch_in, ch_out,
                             kernel_size=kernel_size,
                             ratio_global_in=ratio_global_in,
                             ratio_global_out=ratio_global_out,
                             dtype=dtype)
        return layer(x)

    def apply_conv_block(x, dtype=DEFAULT_DTYPE):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(fmaps, kernel_size=kernel_size, padding='SAME', strides=2, dtype=dtype),
            tf.keras.layers.BatchNormalization(dtype=dtype),
            tf.keras.layers.Activation(tf.nn.relu, dtype=dtype),
        ])
        return model(x)

    def create_model(inputs, dtype=DEFAULT_DTYPE):
        t = inputs
        t = apply_conv_block(t)
        t = apply_ffc_block(t, fmaps, fmaps, dtype=dtype)
        t = apply_ffc_block(t, fmaps, fmaps, dtype=dtype)
        t = apply_ffc_block(t, fmaps, fmaps, dtype=dtype)
        t = tf.keras.layers.Add()(t)
        t = apply_output_block(t)
        model = tf.keras.Model(inputs, t)
        return model

    # Build fp32 model
    fp32_inputs = tf.keras.layers.Input(shape=input_shape)
    fp32_model = create_model(fp32_inputs)
    print('Created fp32 model!')

    # Build fp16 model
    fp16_dtype = tf.keras.mixed_precision.Policy('mixed_float16')
    fp16_inputs = tf.keras.layers.Input(shape=input_shape)
    fp16_model = create_model(fp16_inputs, dtype=fp16_dtype)
    print('Created fp16 model!')

    # Build models and compile evaluations if needed
    fp32_model(x_inputs, training=False)
    fp16_model(x_inputs, training=False)
    print('Built models!')

    # See how much time
    import time
    n_iters = 5
    compute_grads = False

    # Run fp32 model
    start_fp32_time = time.time()
    for i in range(n_iters):
        with tf.GradientTape() as tape:
            outputs = fp32_model(x_inputs, training=True)
            loss = tf.reduce_mean(outputs)
        if compute_grads:
            grads = tape.gradient(loss, fp32_model.trainable_variables)
    total_fp32_time = time.time() - start_fp32_time

    # Run fp16 model
    start_fp16_time = time.time()
    for i in range(n_iters):
        with tf.GradientTape() as tape:
            outputs = fp16_model(x_inputs, training=True)
            loss = tf.reduce_mean(outputs)
        if compute_grads:
            grads = tape.gradient(loss ,fp16_model.trainable_variables)
    total_fp16_time = time.time() - start_fp16_time

    print(f'Time with fp32: {total_fp32_time:3f} sec')
    print(f'Time with mixed precision: {total_fp16_time:3f} sec')


if __name__ == '__main__':
    # prepare_gpu('growth')
    prepare_gpu(memory_limit=7000)
    print('Hello from FFC!')

    tf.keras.backend.set_image_data_format('channels_last')
    print('Image format:', tf.keras.backend.image_data_format())

    run_easy_tests = True
    if run_easy_tests:
        test_spectral_transform()
        test_main_layer()
        test_data_formats()
        test_grad()
    test_fp16_speed()

    """
    Compute grads: true
        NHWC:
        Time with fp32: 3.656294 sec
        Time with mixed precision: 3.179995 sec
        NCHW:
        Time with fp32: 3.842195 sec
        Time with mixed precision: 3.542604 sec
    Compute grads: false
    NHWC:
        Time with fp32: 1.193854 sec
        Time with mixed precision: 0.936233 sec
    NCHW:
        Time with fp32: 1.347324 sec
        Time with mixed precision: 1.325228 sec
    """
