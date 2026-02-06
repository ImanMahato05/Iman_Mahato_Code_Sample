
import pandas as pd
import numpy as np
import pickle
import os
import xlwt
import random


# Define a dataset class for managing train, validation, and test datasets.
class DataSet(object):
    def __init__(self, x_data, y_data, space_dis, time_dis):
        # assert x_data.shape[0] == y_data.shape[0] == space_dis.shape[0] == time_dis.shape[0], 'x,y,distance'
        self._x_data = x_data
        self._y_data = y_data

        if len(space_dis.shape) == 2:
            space_dis = np.reshape(space_dis, [space_dis.shape[0], space_dis.shape[1], 1])
        self._space_dis = space_dis
        if time_dis is not None and len(time_dis.shape) == 2:
            time_dis = np.reshape(time_dis, [time_dis.shape[0], time_dis.shape[1], 1])
        self._time_dis = time_dis
        self._num_examples = x_data.shape[0]
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def x_data(self):
        return self._x_data

    @property
    def y_data(self):
        return self._y_data

    @property
    def space_dis(self):
        return self._space_dis

    @property
    def time_dis(self):
        return self._time_dis

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        # shuffle
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            shuffle = np.arange(self._num_examples)
            np.random.shuffle(shuffle)
            self._x_data = self._x_data[shuffle]
            self._y_data = self._y_data[shuffle]
            self._space_dis = self._space_dis[shuffle]
            self._time_dis = self._time_dis[shuffle]

            
            start = 0
            self._index_in_epoch = batch_size
            assert self._index_in_epoch <= self._num_examples
        end = self._index_in_epoch
        return self._x_data[start:end], self._y_data[start:end], self._space_dis[start:end], self._time_dis[start:end]


def init_dataset_cv(data_path, train_ratio=0.70, validation_ratio=0.15, cv_fold=10, s_each_dir=True, t_cycle=True,
                    log_y=False, normalize_y=True,
                    date_str="", create_force=False, random_fixed=True, seed=10, col_data_x=[], col_data_y=[],
                    col_date=[],
                    col_coordx=[], col_coordy=[], date_numeric=False):
    index_1 = data_path.rfind('/')
    index_2 = data_path.rfind('.')

    dataname = data_path[index_1 + 1:index_2]
    dataset_path = 'Data/dataset/'

    if s_each_dir:
        dataname = dataname + '_sdir'
    if t_cycle:
        dataname = dataname + '_tcyc'

    # If it's not a fixed date, we need to create a new one, ending with the date information.
    if not random_fixed:
        seed = random.randrange(100)
        dataname = dataname + '_' + date_str + '_' + str(seed)
    else:
        dataname = dataname + '_' + str(seed)

    dataname = dataname + '_cv' + str(cv_fold)

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    file_save_path = dataset_path + dataname + '.pckl'

    if create_force:
        if os.path.isfile(file_save_path):
            os.remove(file_save_path)

    if os.path.isfile(file_save_path):
        return open_dataset(file_save_path)
    else:
        all_data = pd.read_csv(data_path, engine='python')
        all_data.dropna(inplace=True)
        all_data = pd.DataFrame(all_data)

        col_names = col_data_x + ['Constant'] + col_data_y
        all_col_names = all_data.columns.values

        # Arguments
        data_varxs_all = all_data[col_data_x]
        data_varxs_all_norm = (data_varxs_all - data_varxs_all.min()) / (data_varxs_all.max() - data_varxs_all.min())
        data_varxs_all_norm['Constant'] = pd.Series(np.ones(len(all_data)), data_varxs_all_norm.index)
        data_varxs_all_norm = data_varxs_all_norm.values
        # dependent variable
        data_vary_all = all_data[col_data_y]
        if log_y:
            data_vary_all_values = np.log2(data_vary_all.values + 1)
        else:
            data_vary_all_values = data_vary_all.values
        miny = np.min(data_vary_all_values)
        maxy = np.max(data_vary_all_values)
        if normalize_y:
            data_vary_all_norm = (data_vary_all_values - miny) / (maxy - miny)
        else:
            data_vary_all_norm = data_vary_all_values

        # Spatial coordinate column
        data_coordx_all = all_data[col_coordx]
        data_coordy_all = all_data[col_coordy]

        s_dis_extent = space_distance_extent_cal(data_coordx_all, data_coordy_all, each_dir=s_each_dir)

        # Divide the data into training set, validation set, and test set.
        all_size = len(all_data)
        shuffle = np.arange(all_size)
        np.random.seed(seed)
        np.random.shuffle(shuffle)

        if cv_fold > 1:
            train_val_size = int(all_size * (train_ratio + validation_ratio))
            test_size = all_size - train_val_size
            train_val_index = shuffle[0:train_val_size]
            test_index = shuffle[train_val_size:]
            train_size = int(train_val_size / cv_fold * (cv_fold - 1))
            val_size = train_val_size - train_size
        else:
            train_val_size = int(all_size * (train_ratio + validation_ratio))
            test_size = all_size - train_val_size
            train_val_index = shuffle[0:train_val_size]
            test_index = shuffle[train_val_size:]
            train_size = int(all_size * train_ratio)
            val_size = train_val_size - train_size

        train_val_datasets = []
        train_datasets = []
        val_datasets = []
        test_datasets = []
        train_coords = []

        for cv_index in range(cv_fold):
            val_index_start = (cv_fold - cv_index - 1) * val_size
            if cv_index == 0:
                val_index_start = train_val_size - val_size
                val_index_end = train_val_size
            else:
                val_index_end = (cv_fold - cv_index) * val_size

            validation_index = train_val_index[val_index_start:val_index_end]
            if val_index_start == 0:
                train_index = train_val_index[val_index_end:]
            elif val_index_end == train_val_size:
                train_index = train_val_index[:val_index_start]
            else:
                train_index = np.hstack((train_val_index[:val_index_start], train_val_index[val_index_end:]))

            data_coordx_train_val = data_coordx_all.iloc[train_val_index]
            data_coordx_train = data_coordx_all.iloc[train_index]
            data_coordx_validation = data_coordx_all.iloc[validation_index]
            data_coordx_test = data_coordx_all.iloc[test_index]

            data_coordy_train_val = data_coordy_all.iloc[train_val_index]
            data_coordy_train = data_coordy_all.iloc[train_index]
            data_coordy_validation = data_coordy_all.iloc[validation_index]
            data_coordy_test = data_coordy_all.iloc[test_index]

            train_coord = [data_coordx_train, data_coordy_train]

            s_dis_frame_train_val = space_distance_cal_quick(data_coordx_train_val.values, data_coordy_train_val.values,
                                                             data_coordx_train.values,
                                                             data_coordy_train.values, each_dir=s_each_dir,
                                                             dataframe=True)
            s_dis_frame_train = space_distance_cal_quick(data_coordx_train.values, data_coordy_train.values,
                                                         data_coordx_train.values,
                                                         data_coordy_train.values, each_dir=s_each_dir, dataframe=True)
            s_dis_frame_validation = space_distance_cal_quick(data_coordx_validation.values,
                                                              data_coordy_validation.values, data_coordx_train.values,
                                                              data_coordy_train.values, each_dir=s_each_dir,
                                                              dataframe=True)
            s_dis_frame_test = space_distance_cal_quick(data_coordx_test.values, data_coordy_test.values,
                                                        data_coordx_train.values, data_coordy_train.values,
                                                        each_dir=s_each_dir, dataframe=True)

            if not s_each_dir:
                s_dis_train_val = np.transpose(
                    ((s_dis_frame_train_val - s_dis_extent[0]) / (s_dis_extent[1] - s_dis_extent[0])).values)
                s_dis_train = np.transpose(
                    ((s_dis_frame_train - s_dis_extent[0]) / (s_dis_extent[1] - s_dis_extent[0])).values)
                s_dis_val = np.transpose(((s_dis_frame_validation - s_dis_extent[0]) / (
                        s_dis_extent[1] - s_dis_extent[0])).values)
                s_dis_test = np.transpose(
                    ((s_dis_frame_test - s_dis_extent[0]) / (s_dis_extent[1] - s_dis_extent[0])).values)
                s_dis_frame_train_val_norm = s_dis_train_val
                s_dis_frame_train_norm = s_dis_train
                s_dis_frame_validation_norm = s_dis_val
                s_dis_frame_test_norm = s_dis_test
            else:
                for i in range(len(s_dis_frame_train_val)):
                    s_dis_train_val = np.transpose(
                        ((s_dis_frame_train_val[i] - s_dis_extent[i][0]) / (
                                    s_dis_extent[i][1] - s_dis_extent[i][0])).values)
                    s_dis_train = np.transpose(
                        ((s_dis_frame_train[i] - s_dis_extent[i][0]) / (
                                s_dis_extent[i][1] - s_dis_extent[i][0])).values)
                    s_dis_val = np.transpose(((s_dis_frame_validation[i] - s_dis_extent[i][0]) / (
                            s_dis_extent[i][1] - s_dis_extent[i][0])).values)
                    s_dis_test = np.transpose(
                        ((s_dis_frame_test[i] - s_dis_extent[i][0]) / (s_dis_extent[i][1] - s_dis_extent[i][0])).values)

                    if i == 0:
                        s_dis_frame_train_val_norm = s_dis_train_val
                        s_dis_frame_train_norm = s_dis_train
                        s_dis_frame_validation_norm = s_dis_val
                        s_dis_frame_test_norm = s_dis_test
                    else:
                        s_dis_frame_train_val_norm = np.dstack((s_dis_frame_train_val_norm, s_dis_train_val))
                        s_dis_frame_train_norm = np.dstack((s_dis_frame_train_norm, s_dis_train))
                        s_dis_frame_validation_norm = np.dstack((s_dis_frame_validation_norm, s_dis_val))
                        s_dis_frame_test_norm = np.dstack((s_dis_frame_test_norm, s_dis_test))

            # Separate the handling of time and distance.
            if len(col_date) > 0 and col_date[0] != 'none':
                data_date_all = all_data[col_date]
                t_dis_extent = time_distance_extent_cal(data_date_all, date_numeric, cycle=t_cycle)
                data_date_train_val = data_date_all.iloc[train_val_index]
                data_date_train = data_date_all.iloc[train_index]
                data_date_validation = data_date_all.iloc[validation_index]
                data_date_test = data_date_all.iloc[test_index]

                train_coord.append(data_date_train)

                t_dis_frame_train_val = time_distance_cal(data_date_train_val, data_date_train, date_numeric,
                                                          cycle=t_cycle)
                t_dis_frame_train = time_distance_cal(data_date_train, data_date_train, date_numeric, cycle=t_cycle)
                t_dis_frame_validation = time_distance_cal(data_date_validation, data_date_train, date_numeric,
                                                           cycle=t_cycle)
                t_dis_frame_test = time_distance_cal(data_date_test, data_date_train, date_numeric, cycle=t_cycle)
                if not t_cycle:
                    t_dis_train_val = np.transpose(
                        ((t_dis_frame_train_val - t_dis_extent[0]) / (t_dis_extent[1] - t_dis_extent[0])).values)
                    t_dis_train = np.transpose(
                        ((t_dis_frame_train - t_dis_extent[0]) / (t_dis_extent[1] - t_dis_extent[0])).values)
                    t_dis_val = np.transpose(((t_dis_frame_validation - t_dis_extent[0]) / (
                            t_dis_extent[1] - t_dis_extent[0])).values)
                    t_dis_test = np.transpose(
                        ((t_dis_frame_test - t_dis_extent[0]) / (t_dis_extent[1] - t_dis_extent[0])).values)

                    t_dis_frame_train_val_norm = t_dis_train_val
                    t_dis_frame_train_norm = t_dis_train
                    t_dis_frame_validation_norm = t_dis_val
                    t_dis_frame_test_norm = t_dis_test
                else:
                    for i in range(len(t_dis_frame_train_val)):
                        t_dis_train_val = np.transpose(
                            ((t_dis_frame_train_val[i] - t_dis_extent[i][0]) / (
                                    t_dis_extent[i][1] - t_dis_extent[i][0])).values)
                        t_dis_train = np.transpose(
                            ((t_dis_frame_train[i] - t_dis_extent[i][0]) / (
                                    t_dis_extent[i][1] - t_dis_extent[i][0])).values)
                        t_dis_val = np.transpose(((t_dis_frame_validation[i] - t_dis_extent[i][0]) / (
                                t_dis_extent[i][1] - t_dis_extent[i][0])).values)
                        t_dis_test = np.transpose(
                            ((t_dis_frame_test[i] - t_dis_extent[i][0]) / (
                                        t_dis_extent[i][1] - t_dis_extent[i][0])).values)

                        if i == 0:
                            t_dis_frame_train_val_norm = t_dis_train_val
                            t_dis_frame_train_norm = t_dis_train
                            t_dis_frame_validation_norm = t_dis_val
                            t_dis_frame_test_norm = t_dis_test
                        else:
                            t_dis_frame_train_val_norm = np.dstack((t_dis_frame_train_val_norm, t_dis_train_val))
                            t_dis_frame_train_norm = np.dstack((t_dis_frame_train_norm, t_dis_train))
                            t_dis_frame_validation_norm = np.dstack((t_dis_frame_validation_norm, t_dis_val))
                            t_dis_frame_test_norm = np.dstack((t_dis_frame_test_norm, t_dis_test))
            else:
                t_dis_frame_train_val_norm = s_dis_frame_train_val_norm
                t_dis_frame_train_norm = s_dis_frame_train_norm
                t_dis_frame_validation_norm = s_dis_frame_validation_norm
                t_dis_frame_test_norm = s_dis_frame_test_norm

            train_val_dataset = DataSet(data_varxs_all_norm[train_val_index], data_vary_all_norm[train_val_index],
                                        s_dis_frame_train_val_norm, t_dis_frame_train_val_norm)

            train_dataset = DataSet(data_varxs_all_norm[train_index], data_vary_all_norm[train_index],
                                    s_dis_frame_train_norm, t_dis_frame_train_norm)

            validation_dataset = DataSet(data_varxs_all_norm[validation_index], data_vary_all_norm[validation_index],
                                         s_dis_frame_validation_norm, t_dis_frame_validation_norm)

            test_dataset = DataSet(data_varxs_all_norm[test_index], data_vary_all_norm[test_index],
                                   s_dis_frame_test_norm, t_dis_frame_test_norm)

            train_val_datasets.append(train_val_dataset)
            train_datasets.append(train_dataset)
            val_datasets.append(validation_dataset)
            test_datasets.append(test_dataset)
            train_coords.append(train_coord)

            excel_save_path = dataset_path + dataname + '_' + str(cv_index + 1) + '.xls'
            # Save the data to Excel
            if os.path.exists(excel_save_path):
                os.remove(excel_save_path)
            workbook = xlwt.Workbook(encoding='utf-8')
            worksheet = workbook.add_sheet('data', cell_overwrite_ok=True)

            workbook = add_data_excel(workbook, name='train',
                                      data_varxs_all_norm=data_varxs_all_norm[train_index],
                                      data_vary_all_norm=data_vary_all_norm[train_index], col_names=col_names,
                                      all_data=np.array(all_data)[train_index], all_col_names=all_col_names,
                                      row_index=0)
            workbook = add_data_excel(workbook, name='validation',
                                      data_varxs_all_norm=data_varxs_all_norm[validation_index],
                                      data_vary_all_norm=data_vary_all_norm[validation_index], col_names=col_names,
                                      all_data=np.array(all_data)[validation_index], all_col_names=all_col_names,
                                      row_index=len(train_index))
            workbook = add_data_excel(workbook, name='test',
                                      data_varxs_all_norm=data_varxs_all_norm[test_index],
                                      data_vary_all_norm=data_vary_all_norm[test_index], col_names=col_names,
                                      all_data=np.array(all_data)[test_index], all_col_names=all_col_names,
                                      row_index=len(train_val_index))

            workbook.add_sheet('result', cell_overwrite_ok=True)

            workbook.save(excel_save_path)

        save_dataset(file_save_path,
                     [train_val_datasets, train_datasets, val_datasets, test_datasets, miny, maxy, dataname,
                      train_coords])
        return train_val_datasets, train_datasets, val_datasets, test_datasets, miny, maxy, dataname, train_coords


# Save spatiotemporal data, independent and dependent variables to Excel
def add_data_excel(workbook, name, data_varxs_all_norm, data_vary_all_norm, col_names, all_data, all_col_names,
                   row_index):
    worksheet = workbook.get_sheet('data')

    worksheet.write(0, 0, 'dataset')

    column_index = 1
    for column in all_col_names:
        worksheet.write(0, column_index, column)
        column_index = column_index + 1

    for column in col_names:
        worksheet.write(0, column_index, column + '_norm')
        column_index = column_index + 1

    start = row_index

    for i in range(data_varxs_all_norm.shape[0]):
        worksheet.write(i + 1 + start, 0, name)
        column_index = 1
        for data_index in range(len(all_col_names)):
            worksheet.write(i + 1 + start, column_index, all_data[i, data_index])
            column_index = column_index + 1

        for data_index in range(data_varxs_all_norm.shape[1]):
            worksheet.write(i + 1 + start, column_index, data_varxs_all_norm[i, data_index])
            column_index = column_index + 1

        worksheet.write(i + 1 + start, column_index, data_vary_all_norm[i, 0])

    return workbook


def time_distance_extent_cal(time_frame, date_numeric, cycle=False):
    if not cycle:
        t_dis_min = 1000000
        t_dis_max = 0
        for i in range(len(time_frame)):
            if date_numeric:
                t_dis_temp = np.abs(time_frame.values[i] - time_frame.values)
            else:
                t_dis_temp = (np.abs((time_frame.values[i] - time_frame.values) / np.timedelta64(1, 'D'))).astype(int)
            t_dis_temp = np.reshape(t_dis_temp, len(t_dis_temp))
            if (np.min(t_dis_temp) < t_dis_min):
                t_dis_min = np.min(t_dis_temp)
            if (np.max(t_dis_temp) > t_dis_max):
                t_dis_max = np.max(t_dis_temp)

        return t_dis_min, t_dis_max
    else:
        t_dis_nc_min = 1000000
        t_dis_nc_max = 0
        t_dis_c_min = 1000000
        t_dis_c_max = 0

        years = time_frame.values.astype('datetime64[Y]').astype(int) + 1970
        month_days = (time_frame.values.astype('datetime64[D]') - time_frame.values.astype('datetime64[Y]')).astype(
            int) + 1

        for i in range(len(time_frame)):
            t_dis_nc = np.abs(years[i] - years)
            t_dis_c = np.abs(month_days[i] - month_days)
            t_dis_nc = np.reshape(t_dis_nc, len(t_dis_nc))
            t_dis_c = np.reshape(t_dis_c, len(t_dis_c))

            if (np.min(t_dis_nc) < t_dis_nc_min):
                t_dis_nc_min = np.min(t_dis_nc)
            if (np.max(t_dis_nc) > t_dis_nc_max):
                t_dis_nc_max = np.max(t_dis_nc)

            if (np.min(t_dis_c) < t_dis_c_min):
                t_dis_c_min = np.min(t_dis_c)
            if (np.max(t_dis_c) > t_dis_c_max):
                t_dis_c_max = np.max(t_dis_c)

        return [t_dis_nc_min, t_dis_nc_max], [t_dis_c_min, t_dis_c_max]


def space_distance_extent_cal(coordx_frame, coordy_frame, each_dir=False):
    if not each_dir:
        s_dis_min = 1000000
        s_dis_max = 0
        for i in range(len(coordx_frame)):
            s_dis_x = coordx_frame.values[i] - coordx_frame.values
            s_dis_y = coordy_frame.values[i] - coordy_frame.values
            s_dis = np.sqrt(s_dis_x * s_dis_x + s_dis_y * s_dis_y)
            s_dis = np.reshape(s_dis, len(s_dis))
            if (np.min(s_dis) < s_dis_min):
                s_dis_min = np.min(s_dis)
            if (np.max(s_dis) > s_dis_max):
                s_dis_max = np.max(s_dis)

        return s_dis_min, s_dis_max
    else:
        s_dis_x_min = 1000000
        s_dis_x_max = 0
        s_dis_y_min = 1000000
        s_dis_y_max = 0

        for i in range(len(coordx_frame)):
            s_dis_x = np.abs(coordx_frame.values[i] - coordx_frame.values)
            s_dis_y = np.abs(coordy_frame.values[i] - coordy_frame.values)
            s_dis_x = np.reshape(s_dis_x, len(s_dis_x))
            s_dis_y = np.reshape(s_dis_y, len(s_dis_y))
            if (np.min(s_dis_x) < s_dis_x_min):
                s_dis_x_min = np.min(s_dis_x)
            if (np.max(s_dis_x) > s_dis_x_max):
                s_dis_x_max = np.max(s_dis_x)

            if (np.min(s_dis_y) < s_dis_y_min):
                s_dis_y_min = np.min(s_dis_y)
            if (np.max(s_dis_y) > s_dis_y_max):
                s_dis_y_max = np.max(s_dis_y)

        return [s_dis_x_min, s_dis_x_max], [s_dis_y_min, s_dis_y_max]


def space_smooth_distance_extent_cal(coordx_frame, coordy_frame, smooth_coordx_frame, smooth_coordy_frame,
                                     each_dir=False):
    if not each_dir:
        s_dis_min = 1000000
        s_dis_max = 0
        for i in range(len(coordx_frame)):
            s_dis_x = coordx_frame.values[i] - smooth_coordx_frame.values
            s_dis_y = coordy_frame.values[i] - smooth_coordy_frame.values
            s_dis = np.sqrt(s_dis_x * s_dis_x + s_dis_y * s_dis_y)
            s_dis = np.reshape(s_dis, len(s_dis))
            if (np.min(s_dis) < s_dis_min):
                s_dis_min = np.min(s_dis)
            if (np.max(s_dis) > s_dis_max):
                s_dis_max = np.max(s_dis)

        return s_dis_min, s_dis_max
    else:
        s_dis_x_min = 1000000
        s_dis_x_max = 0
        s_dis_y_min = 1000000
        s_dis_y_max = 0

        for i in range(len(coordx_frame)):
            s_dis_x = np.abs(coordx_frame.values[i] - smooth_coordx_frame.values)
            s_dis_y = np.abs(coordy_frame.values[i] - smooth_coordy_frame.values)
            s_dis_x = np.reshape(s_dis_x, len(s_dis_x))
            s_dis_y = np.reshape(s_dis_y, len(s_dis_y))
            if (np.min(s_dis_x) < s_dis_x_min):
                s_dis_x_min = np.min(s_dis_x)
            if (np.max(s_dis_x) > s_dis_x_max):
                s_dis_x_max = np.max(s_dis_x)

            if (np.min(s_dis_y) < s_dis_y_min):
                s_dis_y_min = np.min(s_dis_y)
            if (np.max(s_dis_y) > s_dis_y_max):
                s_dis_y_max = np.max(s_dis_y)

        return [s_dis_x_min, s_dis_x_max], [s_dis_y_min, s_dis_y_max]


def time_distance_cal(time_frame, train_time_frame, date_numeric, cycle=False):
    if not cycle:
        t_dis_frame = pd.DataFrame()
        for i in range(len(time_frame)):
            if date_numeric:
                t_dis_temp = np.abs(time_frame.values[i] - train_time_frame.values)
            else:
                t_dis_temp = (np.abs((time_frame.values[i] - train_time_frame.values) / np.timedelta64(1, 'D'))).astype(
                    int)
            t_dis_temp = np.reshape(t_dis_temp, len(t_dis_temp))
            t_dis_frame[str(i + 1)] = pd.Series(t_dis_temp)

        return t_dis_frame
    else:
        t_dis_nc_frame = pd.DataFrame()
        t_dis_c_frame = pd.DataFrame()

        years = time_frame.values.astype('datetime64[Y]').astype(int) + 1970
        month_days = (time_frame.values.astype('datetime64[D]') - time_frame.values.astype('datetime64[Y]')).astype(
            int) + 1

        train_years = train_time_frame.values.astype('datetime64[Y]').astype(int) + 1970
        train_month_days = (train_time_frame.values.astype('datetime64[D]') - train_time_frame.values.astype(
            'datetime64[Y]')).astype(
            int) + 1

        for i in range(len(time_frame)):
            t_dis_nc = np.abs(years[i] - train_years)
            t_dis_c = np.abs(month_days[i] - train_month_days)
            t_dis_nc = np.reshape(t_dis_nc, len(t_dis_nc))
            t_dis_c = np.reshape(t_dis_c, len(t_dis_c))

            t_dis_nc_frame[str(i + 1)] = pd.Series(t_dis_nc)
            t_dis_c_frame[str(i + 1)] = pd.Series(t_dis_c)

        return t_dis_nc_frame, t_dis_c_frame


def space_distance_cal_quick(coordx, coordy, train_coordx, train_coordy, each_dir=False, dataframe=False):
    len_pred = len(coordx)
    len_train = len(train_coordx)

    coordx = np.expand_dims(np.reshape(coordx, len_pred), axis=0)
    coordy = np.expand_dims(np.reshape(coordy, len_pred), axis=0)

    coordx_all = np.repeat(coordx, len_train, axis=0)
    coordy_all = np.repeat(coordy, len_train, axis=0)

    train_coordx_all = np.repeat(train_coordx, len_pred, axis=1)
    train_coordy_all = np.repeat(train_coordy, len_pred, axis=1)

    if not each_dir:
        s_dis_x = coordx_all - train_coordx_all
        s_dis_y = coordy_all - train_coordy_all
        s_dis = np.sqrt(s_dis_x * s_dis_x + s_dis_y * s_dis_y)
        if dataframe:
            return pd.DataFrame(s_dis)
        else:
            return s_dis

    else:
        s_dis_x = np.abs(coordx_all - train_coordx_all)
        s_dis_y = np.abs(coordy_all - train_coordy_all)

        if dataframe:
            return pd.DataFrame(s_dis_x), pd.DataFrame(s_dis_y)
        else:
            return s_dis_x, s_dis_y


def space_distance_cal(coordx_frame, coordy_frame, train_coordx_frame, train_coordy_frame, each_dir=False):
    if not each_dir:
        s_dis_frame = pd.DataFrame()
        for i in range(len(coordx_frame)):
            s_dis_x = coordx_frame.values[i] - train_coordx_frame.values
            s_dis_y = coordy_frame.values[i] - train_coordy_frame.values
            s_dis = np.sqrt(s_dis_x * s_dis_x + s_dis_y * s_dis_y)
            s_dis = np.reshape(s_dis, len(s_dis))
            s_dis_frame[str(i + 1)] = pd.Series(s_dis)

        return s_dis_frame
    else:
        s_dis_x_frame = pd.DataFrame()
        s_dis_y_frame = pd.DataFrame()
        for i in range(len(coordx_frame)):
            s_dis_x = np.abs(coordx_frame.values[i] - train_coordx_frame.values)
            s_dis_y = np.abs(coordy_frame.values[i] - train_coordy_frame.values)
            s_dis_x = np.reshape(s_dis_x, len(s_dis_x))
            s_dis_y = np.reshape(s_dis_y, len(s_dis_y))
            s_dis_x_frame[str(i + 1)] = pd.Series(s_dis_x)
            s_dis_y_frame[str(i + 1)] = pd.Series(s_dis_y)

        return s_dis_x_frame, s_dis_y_frame


def space_smooth_distance_cal(smooth_coordx_frame, smooth_coordy_frame, coordx_frame, coordy_frame, each_dir=False):
    if not each_dir:
        s_dis_frame = pd.DataFrame()
        for i in range(len(coordx_frame)):
            s_dis_x = coordx_frame.values[i] - smooth_coordx_frame.values
            s_dis_y = coordy_frame.values[i] - smooth_coordy_frame.values
            s_dis = np.sqrt(s_dis_x * s_dis_x + s_dis_y * s_dis_y)
            s_dis = np.reshape(s_dis, len(s_dis))
            s_dis_frame[str(i + 1)] = pd.Series(s_dis)

        return s_dis_frame
    else:
        s_dis_x_frame = pd.DataFrame()
        s_dis_y_frame = pd.DataFrame()
        for i in range(len(coordx_frame)):
            s_dis_x = np.abs(coordx_frame.values[i] - smooth_coordx_frame.values)
            s_dis_y = np.abs(coordy_frame.values[i] - smooth_coordy_frame.values)
            s_dis_x = np.reshape(s_dis_x, len(s_dis_x))
            s_dis_y = np.reshape(s_dis_y, len(s_dis_y))
            s_dis_x_frame[str(i + 1)] = pd.Series(s_dis_x)
            s_dis_y_frame[str(i + 1)] = pd.Series(s_dis_y)

        return s_dis_x_frame, s_dis_y_frame


def save_dataset(dataname, dataset_list):
    f = open(dataname, 'wb')
    pickle.dump(dataset_list, f)
    f.close()


def open_dataset(dataname):
    f = open(dataname, 'rb')
    train_val_datasets, train_datasets, val_datasets, test_datasets, miny, maxy, dataname, train_coords = pickle.load(f)
    f.close()
    return train_val_datasets, train_datasets, val_datasets, test_datasets, miny, maxy, dataname, train_coords
