# -*- coding:utf-8 -*-
"""

Author:
    Weichen Shen,weichenswc@163.com

"""
import numpy as np
import torch
import os


def concat_fun(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return torch.cat(inputs, dim=axis)


def slice_arrays(arrays, start=None, stop=None):
    """Slice an array or list of arrays.

    This takes an array-like, or a list of
    array-likes, and outputs:
        - arrays[start:stop] if `arrays` is an array-like
        - [x[start:stop] for x in arrays] if `arrays` is a list

    Can also work on list/array of indices: `slice_arrays(x, indices)`

    Arguments:
        arrays: Single array or list of arrays.
        start: can be an integer index (start index)
            or a list/array of indices
        stop: integer (stop index); should be None if
            `start` was a list.

    Returns:
        A slice of the array(s).

    Raises:
        ValueError: If the value of start is a list and stop is not None.
    """

    if arrays is None:
        return [None]

    if isinstance(arrays, np.ndarray):
        arrays = [arrays]

    if isinstance(start, list) and stop is not None:
        raise ValueError('The stop argument has to be None if the value of start '
                         'is a list.')
    elif isinstance(arrays, list):
        if hasattr(start, '__len__'):
            # hdf5 datasets only support list objects as indices
            if hasattr(start, 'shape'):
                start = start.tolist()
            return [None if x is None else x[start] for x in arrays]
        else:
            if len(arrays) == 1:
                return arrays[0][start:stop]
            return [None if x is None else x[start:stop] for x in arrays]
    else:
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return arrays[start]
        elif hasattr(start, '__getitem__'):
            return arrays[start:stop]
        else:
            return [None]


def save_model(model, model_dir, epoch=None):
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + '_dhfm.pt')
    with open(file_name, 'wb') as f:
        torch.save(model, f)


def load_model(model_dir, epoch=None):
    if not model_dir:
        return
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + '_dhfm.pt')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(file_name):
        return
    with open(file_name, 'rb') as f:
        model = torch.load(f)
    return model

def masked_MAPE(v, v_, axis=None):
    '''
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAPE averages on all elements of input.
    '''
    mask = (v == 0)
    percentage = np.abs(v_ - v) / np.abs(v)
    if np.any(mask):
        masked_array = np.ma.masked_array(percentage, mask=mask)  # mask the dividing-zero as invalid
        result = masked_array.mean(axis=axis)
        if isinstance(result, np.ma.MaskedArray):
            return result.filled(np.nan)
        else:
            return result
    return np.mean(percentage, axis).astype(np.float64)

"""
original
def MAPE(v, v_, axis=None):
    '''
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAPE averages on all elements of input.
    '''
    mape = (np.abs(v_ - v) / np.abs(v)+1e-5).astype(np.float64)
    mape = np.where(mape > 5, 5, mape)
    return np.mean(mape, axis)

"""

def MAPE(v, v_, axis=None):
    '''
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: float, MAPE averages on all elements of input.
    '''
    mape = (np.abs(v_ - v) / (np.abs(v) + 1e-5)).astype(np.float64)
    mape = np.where(mape > 5, 5, mape)  # clip extreme values
    return np.mean(mape, axis)


#def MAPE(true, pred):
#    return np.mean(np.abs((pred - true) / (true+1e-5)))

def smape(P, A):
    nz = np.where(A > 0)
    Pz = P[nz]
    Az = A[nz]

    return np.mean(2 * np.abs(Az - Pz) / (np.abs(Az) + np.abs(Pz)))


def R2(y, y_hat, axis=None, eps=1e-12):
    """
    R^2 score for arrays shaped like [count, time_step, node] (or compatible).
    axis=None -> global scalar R2 over all elements.
    axis can be int or tuple of ints: reduce over those axes, keeping the others.
    """
    y = np.asarray(y, dtype=np.float64)
    y_hat = np.asarray(y_hat, dtype=np.float64)

    # residual sum of squares
    ss_res = np.sum((y - y_hat) ** 2, axis=axis)

    # total sum of squares around mean of y along the same reduction axis
    y_mean = np.mean(y, axis=axis, keepdims=True)
    ss_tot = np.sum((y - y_mean) ** 2, axis=axis)

    # Avoid division by zero (constant targets)
    denom = ss_tot + eps
    r2 = 1.0 - (ss_res / denom)

    # If ss_tot is truly ~0, R2 is not well-defined; mark as nan
    # (Optional) If you want 0.0 instead, replace np.nan with 0.0
    if np.isscalar(ss_tot):
        if ss_tot < eps:
            return np.nan
        return float(r2)

    r2 = np.where(ss_tot < eps, np.nan, r2)
    return r2.astype(np.float64)

def RSE(v, v_, axis=None, eps=1e-12):
    '''
    Relative squared error (rooted):
        sqrt( sum((v_ - v)^2) / sum((v - mean(v))^2) )
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: float, RSE on all elements of input (or reduced by axis).
    '''
    v = np.asarray(v, dtype=np.float64)
    v_ = np.asarray(v_, dtype=np.float64)

    v_mean = np.mean(v, axis=axis, keepdims=True)
    num = np.sum((v_ - v) ** 2, axis=axis)
    denom = np.sum((v - v_mean) ** 2, axis=axis)
    return np.sqrt(num / (denom + eps)).astype(np.float64)

def RMSE(v, v_, axis=None):
    '''
    Mean squared error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, RMSE averages on all elements of input.
    '''
    return np.sqrt(np.mean((v_ - v) ** 2, axis)).astype(np.float64)


def MAE(v, v_, axis=None):
    '''
    Mean absolute error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAE averages on all elements of input.
    '''
    return np.mean(np.abs(v_ - v), axis).astype(np.float64)


def evaluate(y, y_hat, by_step=False, by_node=False):
    '''
    :param y: array in shape of [count, time_step, node].
    :param y_hat: in same shape with y.
    :param by_step: evaluate by time_step dim.
    :param by_node: evaluate by node dim.
    :return: array of mape, mae and rmse.
    '''
    if not by_step and not by_node:
        return MAPE(y, y_hat), MAE(y, y_hat), RMSE(y, y_hat), R2(y, y_hat), RSE(y, y_hat)
    if by_step and by_node:
        return MAPE(y, y_hat, axis=0), MAE(y, y_hat, axis=0), RMSE(y, y_hat, axis=0), R2(y, y_hat, axis=0)
    if by_step:
        return MAPE(y, y_hat, axis=(0, 2)), MAE(y, y_hat, axis=(0, 2)), RMSE(y, y_hat, axis=(0, 2)), R2(y, y_hat, axis=(0, 2))
    if by_node:
        return MAPE(y, y_hat, axis=(0, 1)), MAE(y, y_hat, axis=(0, 1)), RMSE(y, y_hat, axis=(0, 1)), R2(y, y_hat, axis=(0, 1))


def save_model_ts(model, path, epoch):
    if not os.path.exists(path):
        os.makedirs(path)
    filename = 'epoch_{}.pth'.format(epoch)
    f = os.path.join(path, filename)
    # Save state_dict instead of the entire model
    torch.save(model.state_dict(), f)

def load_model_ts(model, path, epoch):
    """Load state dict into an existing model instance"""
    filename = 'epoch_{}.pth'.format(epoch)
    f = os.path.join(path, filename)
    model.load_state_dict(torch.load(f))
    return model
