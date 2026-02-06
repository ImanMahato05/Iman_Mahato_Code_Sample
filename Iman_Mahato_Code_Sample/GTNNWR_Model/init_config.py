from configparser import ConfigParser
import tensorflow as tf
import json
import codecs



class ConfigSetting(object):
    def __init__(self, data_setting_path, model_setting_path):
        self._data_setting_path = data_setting_path
        self._model_setting_path = model_setting_path
        self.cp_data = ConfigParser()
        self.cp_data.read(data_setting_path)
        self.cp_model = ConfigParser()
        self.cp_model.read(model_setting_path)

        # global para
        self._diff_weight = self.cp_data.getboolean("global_setting", "diff_weight")
        self._batch_norm = self.cp_data.getboolean("global_setting", "batch_norm")
        self._create_force = self.cp_data.getboolean("global_setting", "create_force")
        self._random_fixed = self.cp_data.getboolean("global_setting", "random_fixed")
        self._base_path = self.cp_data.get("global_setting", "base_path")
        self._log_y = self.cp_data.getboolean("global_setting", "log_y")
        self._simple_stnn = self.cp_data.getboolean("global_setting", "simple_stnn")
        self._stannr = self.cp_data.getboolean("global_setting", "stannr")
        self._seed = self.cp_data.getint("global_setting", "seed")
        self._iter_num = self.cp_data.getint("global_setting", "iter_num")
        self._stop_num = self.cp_data.getint("global_setting", "stop_num")
        self._stop_iter = self.cp_data.getint("global_setting", "stop_iter")
        self._train_r2_cri = self.cp_data.getfloat("global_setting", "train_r2_cri")
        self._test_r2_cri = self.cp_data.getfloat("global_setting", "test_r2_cri")
        self._log_delete = self.cp_data.getint("global_setting", "log_delete")
        self._test_model = self.cp_data.getint("global_setting", "test_model")
        self._cv_fold = self.cp_data.getint("global_setting", "cv_fold")

        # data para
        self._col_data_x = self.cp_data.get("data_setting", "col_data_x").split(',')
        self._col_data_y = self.cp_data.get("data_setting", "col_data_y").split(',')
        self._col_coordx = self.cp_data.get("data_setting", "col_coordx").split(',')
        self._col_coordy = self.cp_data.get("data_setting", "col_coordy").split(',')
        self._col_coordt = self.cp_data.get("data_setting", "col_coordt").split(',')
        self._seasons = self.cp_data.get("data_setting", "seasons").split(',')

        self._models = self.cp_data.get("data_setting", "models").split(',')
        self._simple_stnns_str = self.cp_data.get("data_setting", "simple_stnns").split(',')
        self._simple_stnns = []
        for i in range(len(self._simple_stnns_str)):
            if self._simple_stnns_str[i] == 'True':
                self._simple_stnns.append(True)
            else:
                self._simple_stnns.append(False)
        self._stannrs_str = self.cp_data.get("data_setting", "stannrs").split(',')
        self._stannrs = []
        for i in range(len(self._stannrs_str)):
            if self._stannrs_str[i] == 'True':
                self._stannrs.append(True)
            else:
                self._stannrs.append(False)
        self._simple_gtwnns_str = self.cp_data.get("data_setting", "simple_gtwnns").split(',')
        self._simple_gtwnns = []
        for i in range(len(self._simple_gtwnns_str)):
            if self._simple_gtwnns_str[i] == 'True':
                self._simple_gtwnns.append(True)
            else:
                self._simple_gtwnns.append(False)
        self._datafile = self.cp_data.get("data_setting", "datafile")
        self._log_y = self.cp_data.getboolean("data_setting", "log_y")
        self._normalize_y = self.cp_data.getboolean("data_setting", "normalize_y")
        self._train_ratio = self.cp_data.getfloat("data_setting", "train_ratio")
        self._validation_ratio = self.cp_data.getfloat("data_setting", "validation_ratio")
        self._st_weight_init = self.cp_data.get("data_setting", "st_weight_init")
        self._gtw_weight_init = self.cp_data.get("data_setting", "gtw_weight_init")

        self._epochs = self.cp_data.getint("data_setting", "epochs")
        self._start_lr = self.cp_data.getfloat("data_setting", "start_lr")
        self._max_lr = self.cp_data.getfloat("data_setting", "max_lr")
        self._total_up_steps = self.cp_data.getint("data_setting", "total_up_steps")
        self._up_decay_steps = self.cp_data.getint("data_setting", "up_decay_steps")
        self._maintain_maxlr_steps = self.cp_data.getint("data_setting", "maintain_maxlr_steps")
        self._delay_steps = self.cp_data.getint("data_setting", "delay_steps")
        self._delay_rate = self.cp_data.getfloat("data_setting", "delay_rate")
        self._keep_prob_ratio = self.cp_data.getfloat("data_setting", "keep_prob_ratio")

        self._val_early_stopping = self.cp_data.getboolean("data_setting", "val_early_stopping")
        self._val_early_stopping_begin_step = self.cp_data.getint("data_setting", "val_early_stopping_begin_step")
        self._model_comparison_criterion = self.cp_data.getfloat("data_setting", "model_comparison_criterion")

        # model structure
        self._snn_hidden_layer_count = self.cp_data.getint("structure", "snn_hidden_layer_count")
        self._snn_neural_sizes = json.loads(self.cp_data.get("structure", "snn_neural_sizes"))
        self._snn_output_size = self.cp_data.getint("structure", "snn_output_size")

        self._tnn_hidden_layer_count = self.cp_data.getint("structure", "tnn_hidden_layer_count")
        self._tnn_neural_sizes = json.loads(self.cp_data.get("structure", "tnn_neural_sizes"))
        self._tnn_output_size = self.cp_data.getint("structure", "tnn_output_size")

        self._stnn_hidden_layer_count = self.cp_data.getint("structure", "stnn_hidden_layer_count")
        self._stnn_neural_sizes = json.loads(self.cp_data.get("structure", "stnn_neural_sizes"))
        self._stnn_output_size = self.cp_data.getint("structure", "stnn_output_size")

        self._gtwnn_factor = self.cp_data.getint("structure", "gtwnn_factor")
        self._gtwnn_hidden_node_limit = self.cp_data.getint("structure", "gtwnn_hidden_node_limit")
        self._gtwnn_max_layer_count = self.cp_data.getint("structure", "gtwnn_max_layer_count")

        # CNN
        self._kernel_size = self.cp_data.getint("structure", "kernel_size")
        self._kernel_num = self.cp_data.getint("structure", "kernel_num")
        self._smooth_coords_path = self.cp_data.get("structure", "smooth_coords_path")
        self._x_len = self.cp_data.getint("structure", "x_len")
        self._cnngtwnn_factor = self.cp_data.getint("structure", "cnngtwnn_factor")
        self._cnngtwnn_hidden_node_limit = self.cp_data.getint("structure", "cnngtwnn_hidden_node_limit")
        self._cnngtwnn_max_layer_count = self.cp_data.getint("structure", "cnngtwnn_max_layer_count")

    def get_global_para(self):
        return self._diff_weight, self._batch_norm, self._create_force, self._random_fixed, self._base_path, self._log_y, \
               self._simple_stnn, self._stannr, self._seed, self._iter_num, \
               self._stop_num, self._stop_iter, self._train_r2_cri, self._test_r2_cri, self._log_delete, self._test_model, self._cv_fold

    def get_data_para(self):
        return self._col_data_x, self._col_data_y, self._col_coordx, self._col_coordy, self._col_coordt, self._seasons, self._models, self._simple_stnns, self._stannrs, self._simple_gtwnns, \
               self._datafile, self._log_y, self._normalize_y, self._train_ratio, self._validation_ratio, self._st_weight_init, \
               self._gtw_weight_init, self._epochs, self._start_lr, self._max_lr, self._total_up_steps, self._up_decay_steps, self._maintain_maxlr_steps, \
               self._delay_steps, self._delay_rate, self._keep_prob_ratio, self._val_early_stopping, self._val_early_stopping_begin_step, \
               self._model_comparison_criterion

    def get_model_structure(self):
        return self._snn_hidden_layer_count, self._snn_neural_sizes, self._snn_output_size, \
               self._tnn_hidden_layer_count, self._tnn_neural_sizes, self._tnn_output_size, \
               self._stnn_hidden_layer_count, self._stnn_neural_sizes, self._stnn_output_size, \
               self._gtwnn_factor, self._gtwnn_hidden_node_limit, self._gtwnn_max_layer_count, \
               self._kernel_size, self._kernel_num, self._smooth_coords_path, self._x_len, \
               self._cnngtwnn_factor, self._cnngtwnn_hidden_node_limit, self._cnngtwnn_max_layer_count

    def get_model_para(self, model_name):
        self._no_space = self.cp_model.getboolean(model_name, "no_space")
        self._s_no_network = self.cp_model.getboolean(model_name, "s_no_network")
        self._no_time = self.cp_model.getboolean(model_name, "no_time")
        self._t_no_network = self.cp_model.getboolean(model_name, "t_no_network")
        self._st_no_network = self.cp_model.getboolean(model_name, "st_no_network")
        self._s_each_dir = self.cp_model.getboolean(model_name, "s_each_dir")
        self._t_cycle = self.cp_model.getboolean(model_name, "t_cycle")

        if self.cp_model.get(model_name, "s_activate_fun") == 'identity':
            self._s_activate_fun = tf.identity
        elif self.cp_model.get(model_name, "s_activate_fun") == 'relu':
            self._s_activate_fun = tf.nn.relu
        elif self.cp_model.get(model_name, "s_activate_fun") == 'prelu':
            self._s_activate_fun = 'prelu'
        else:
            self._s_activate_fun = ''

        if self.cp_model.get(model_name, "t_activate_fun") == 'identity':
            self._t_activate_fun = tf.identity
        elif self.cp_model.get(model_name, "t_activate_fun") == 'relu':
            self._t_activate_fun = tf.nn.relu
        elif self.cp_model.get(model_name, "t_activate_fun") == 'prelu':
            self._t_activate_fun = 'prelu'
        else:
            self._t_activate_fun = ''

        if self.cp_model.get(model_name, "st_activate_fun") == 'identity':
            self._st_activate_fun = tf.identity
        elif self.cp_model.get(model_name, "st_activate_fun") == 'relu':
            self._st_activate_fun = tf.nn.relu
        elif self.cp_model.get(model_name, "st_activate_fun") == 'prelu':
            self._st_activate_fun = 'prelu'
        else:
            self._st_activate_fun = ''

        if self.cp_model.get(model_name, "gtw_activate_fun") == 'identity':
            self._gtw_activate_fun = tf.identity
        elif self.cp_model.get(model_name, "gtw_activate_fun") == 'relu':
            self._gtw_activate_fun = tf.nn.relu
        elif self.cp_model.get(model_name, "gtw_activate_fun") == 'prelu':
            self._gtw_activate_fun = 'prelu'
        else:
            self._gtw_activate_fun = ''

        self._no_cnn = self.cp_model.getboolean(model_name, "no_cnn")
        self._dataset_path = self.cp_model.get(model_name, "dataset_path")

        return self._no_space, self._s_no_network, self._no_time, self._t_no_network, self._st_no_network, self._s_each_dir, \
               self._t_cycle, self._s_activate_fun, self._t_activate_fun, self._st_activate_fun, self._gtw_activate_fun, self._no_cnn, self._dataset_path
