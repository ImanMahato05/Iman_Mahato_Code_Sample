import tensorflow as tf
import numpy as np
from scipy.stats.stats import pearsonr
from tensorflow.contrib import layers
import math


def para_relu(_x, var_name="prelu", init_value=0.1):
    _alpha = tf.get_variable(var_name, shape=_x.get_shape()[-1], dtype=_x.dtype,
                             initializer=tf.constant_initializer(init_value))
    return tf.maximum(_alpha * _x, _x)


# Custom learning rate function
def exponential_decay_norm(start_lr, max_lr, global_step, decay_steps, decay_rate, total_up_steps, up_decay_steps, maintain_maxlr_steps,
                           staircase=False, name=None):
    if global_step is None:
        raise ValueError("global_step is required for exponential_decay.")
    with tf.name_scope(name, "ExponentialDecay",
                       [start_lr, max_lr, global_step,
                        decay_steps, decay_rate, total_up_steps, up_decay_steps, maintain_maxlr_steps]) as name:
        start_lr = tf.convert_to_tensor(start_lr, name="start_lr")
        dtype = start_lr.dtype
        max_lr = tf.cast(max_lr, dtype)
        global_step = tf.cast(global_step, dtype)
        decay_steps = tf.cast(decay_steps, dtype)
        decay_rate = tf.cast(decay_rate, dtype)
        total_up_steps = tf.cast(total_up_steps, dtype)
        up_decay_steps = tf.cast(up_decay_steps, dtype)
        maintain_maxlr_steps = tf.cast(maintain_maxlr_steps, dtype)

        up_rate = (max_lr - start_lr) / tf.floor(total_up_steps / up_decay_steps)

        flag_up = tf.cond(tf.greater_equal(total_up_steps, global_step), lambda: True, lambda: False)
        flag_maintain = tf.cond(tf.greater_equal(total_up_steps + maintain_maxlr_steps, global_step), lambda: True, lambda: False)

        p = tf.cond(flag_up, lambda: tf.floor(global_step / up_decay_steps) * up_rate,
                    lambda: tf.cond(flag_maintain, lambda: max_lr,
                                    lambda: tf.floor((global_step - total_up_steps - maintain_maxlr_steps) / decay_steps)))
        re = tf.cond(flag_up, lambda: tf.add(start_lr, p),
                     lambda: tf.cond(flag_maintain, lambda: max_lr, lambda: tf.multiply(max_lr, tf.pow(decay_rate, p)),
                                     name=name))

        return re


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


# Define Variable Functions (weights and bias)
def init_weight_std(var_name, shape, st_dev=0.1):
    return tf.Variable(initial_value=tf.random_normal(shape, stddev=st_dev), name=var_name)


def init_weight_he(var_name, shape, last_layer_num, factor=2.0):
    return tf.Variable(initial_value=tf.random_normal(shape) * np.sqrt(factor / last_layer_num), name=var_name)


def init_bias(name, shape):
    return tf.Variable(initial_value=tf.zeros(shape), name=name)


# Create a fully connected layer:
def fully_connected(layer_name, input_layer, weights, biases, batch_norm=False, is_training=False, summary=False):
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        if summary:
            with tf.name_scope('weights'):
                variable_summaries(weights)
            with tf.name_scope('biases'):
                variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            layer = tf.add(tf.matmul(input_layer, weights), biases)

            if batch_norm:
                # If fused is true, an error will occur when setting is_training=true.
                layer = layers.batch_norm(layer, center=True, scale=True,
                                          is_training=is_training, fused=False, decay=0.999)

            if summary:
                tf.summary.histogram('pre_activations', layer)
    return layer


# Ordinary Linear Regression
def linear(x_data, y_data):
    linear_beta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(x_data), x_data)), tf.transpose(x_data)),
                            y_data)
    return linear_beta


# Calculate r2 based on y_actual and y_pre
def getR2(y_actual, y_pre):
    y_actual = np.reshape(y_actual, len(y_actual))
    y_pre = np.reshape(y_pre, len(y_pre))
    r = pearsonr(y_actual, y_pre)[0]
    re = r ** 2
    return re


# Algorithm for Multiple Linear Regression
def coeff_r2_tensor_cal(ydata, yhat):
    total_error = tf.reduce_sum(tf.square(ydata - tf.reduce_mean(ydata)))
    unexplained_error = tf.reduce_sum(tf.square(ydata - yhat))
    R_squared = 1 - tf.div(unexplained_error, total_error)
    return R_squared


# Algorithm for Multiple Linear Regression
def coeff_r2_adjusted_tensor_cal(ydata, yhat, d1, d2):
    unexplained_error = tf.div(tf.reduce_sum(tf.square(ydata - yhat)), d1)
    total_error = tf.div(tf.reduce_sum(tf.square(ydata - tf.reduce_mean(ydata))), d2)
    R_squared = 1 - tf.div(unexplained_error, total_error)
    return R_squared


# Pearson's r squared
def pearson_r2_tensor_cal(ydata, yhat):
    ydatam = ydata - tf.reduce_mean(ydata)
    yhatm = yhat - tf.reduce_mean(yhat)
    r_num = tf.reduce_sum(ydatam * yhatm)
    r_den = tf.sqrt(tf.reduce_sum(tf.square(ydatam)) * tf.reduce_sum(tf.square(yhatm)))
    r = r_num / r_den
    r = tf.maximum(tf.minimum(r, 1.0), -1.0)
    r2 = tf.square(r)

    return r2


def hidden_layers(sample_size, factor, hidden_node_limit, max_layer_count):
    layer_count = 1
    layer_nodes = [2 ** math.floor(math.log2(sample_size / factor))]
    sample_size = sample_size / factor
    while sample_size / factor >= hidden_node_limit and layer_count < max_layer_count - 1:
        layer_nodes.append(2 ** math.floor(math.log2(sample_size / factor)))
        layer_count = layer_count + 1
        sample_size = sample_size / factor

    layer_nodes.append(2 ** math.floor(math.log2(sample_size / factor)))
    layer_count = layer_count + 1

    return layer_count, layer_nodes


def hidden_layers_2(sample_size, factor, layer_count):
    layer_node = 2 ** math.floor(math.log2(sample_size / factor))
    layer_nodes = np.ones(layer_count, np.int) * layer_node
    return layer_count, layer_nodes


def add_result_excel(workbook, readbook, name, maxy, miny, yhat, gtbeta, gtweight, index, col_data_x, col_data_y, add_col):

    worksheet = workbook.get_sheet('data')
    rs = readbook.sheet_by_name('data')

    yhat_init = yhat * (maxy - miny) + miny
    yhat_list = np.ndarray.tolist(yhat)
    yhat_init_list = np.ndarray.tolist(yhat_init)
    gtbeta = np.ndarray.tolist(gtbeta)
    gtweight = np.ndarray.tolist(gtweight)

    start_col = rs.ncols - 1 + add_col

    worksheet.write(0, start_col + 1, col_data_y[0] + '_pre_norm')
    worksheet.write(0, start_col + 2, col_data_y[0] + '_pre')
    worksheet.write(0, start_col + 3, 'abs_error')
    worksheet.write(0, start_col + 4, 'rel_error')

    for x_i in range(len(col_data_x)):
        worksheet.write(0, start_col + 5 + x_i, col_data_x[x_i] + '_beta')
    worksheet.write(0, start_col + 5 + len(col_data_x), 'constant_beta')

    for x_i in range(len(col_data_x)):
        worksheet.write(0, start_col + 6 + len(col_data_x) + x_i, col_data_x[x_i] + '_weight')
    worksheet.write(0, start_col + 6 + len(col_data_x) * 2, 'constant_weight')

    ycol_index = -1
    for i in range(rs.ncols):
        if rs.cell_value(0, i) == col_data_y[0]:
            ycol_index = i
            break

    for i in range(yhat.shape[0]):
        worksheet.write(i + 1 + index, start_col + 1, yhat_list[i])
        worksheet.write(i + 1 + index, start_col + 2, yhat_init_list[i])
        worksheet.write(i + 1 + index, start_col + 3, abs(yhat_init_list[i] - rs.cell(i + 1 + index, ycol_index).value))
        worksheet.write(i + 1 + index, start_col + 4, abs(yhat_init_list[i] - rs.cell(i + 1 + index, ycol_index).value) / max(rs.cell(i + 1 + index, ycol_index).value, 0.01) * 100.0)

        for x_i in range(len(col_data_x)):
            worksheet.write(i + 1 + index, start_col + 5 + x_i, gtbeta[i][x_i])
        worksheet.write(i + 1 + index, start_col + 5 + len(col_data_x), gtbeta[i][len(col_data_x)])

        for x_i in range(len(col_data_x)):
            worksheet.write(i + 1 + index, start_col + 6 + len(col_data_x) + x_i, gtweight[i][x_i])
        worksheet.write(i + 1 + index, start_col + 6 + len(col_data_x) * 2, gtweight[i][len(col_data_x)])

    return workbook