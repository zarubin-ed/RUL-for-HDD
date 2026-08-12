from typing import Optional, Tuple

import numpy as np
import xgboost as xgb

# --------------------------
# Sample from notebook
# --------------------------
# def censored_mse(failed : pd.Series, y_pred : pd.Series, y_true : pd.Series, sample_weight : Optional[np.ndarray] =None):
#     failed = np.asarray(failed, dtype=float)
#     y_pred = np.asarray(y_pred, dtype=float)
#     y_true = np.asarray(y_true, dtype=float)
#     ans = np.zeros_like(y_pred)
#     mask_failed = failed == 1
#     ans[mask_failed] = np.power(y_pred[mask_failed] - y_true[mask_failed], 2)
#     ans[~mask_failed] = np.where(y_pred[~mask_failed] >= y_true[~mask_failed], 0, np.power(y_pred[~mask_failed] - y_true[~mask_failed], 2))
#     if sample_weight is None:
#         return ans.mean()
#     return (ans * sample_weight).sum() / sample_weight.sum()
# --------------------------



def evaluate(y_pred : np.ndarray, y_true: np.ndarray, weights: Optional[np.ndarray]=None):
    ans = np.zeros_like(y_pred)
    mask_failed = y_true != 60
    ans[mask_failed] = np.power(y_pred[mask_failed] - y_true[mask_failed], 2)
    ans[~mask_failed] = np.where(y_pred[~mask_failed] >= y_true[~mask_failed], 0,
                                 np.power(y_pred[~mask_failed] - y_true[~mask_failed], 2))

    if weights is not None:
        w = np.asarray(weights)
        return (ans * w).sum(), w.sum()

    return ans.sum(), len(ans)

def gradient(y_pred: np.ndarray, y_true: np.ndarray, weights: Optional[np.ndarray]=None) -> np.ndarray:
    ans = np.zeros_like(y_pred)
    mask_failed = y_true != 60

    ans[mask_failed] = 2 * (y_pred[mask_failed] - y_true[mask_failed])
    ans[~mask_failed] = np.where(y_pred[~mask_failed] >= y_true[~mask_failed], 0,
                                 2 * (y_pred[~mask_failed] - y_true[~mask_failed]))

    if weights is None:
        return ans
    else:
        assert len(weights) == len(y_pred)

    return ans * weights

def hessian(y_pred: np.ndarray, y_true: np.ndarray, weights: Optional[np.ndarray]=None) -> np.ndarray:
    ans = np.zeros_like(y_pred)
    mask_failed = y_true != 60

    ans[mask_failed] = 2
    ans[~mask_failed] = np.where(y_pred[~mask_failed] >= y_true[~mask_failed], 0, 2)

    if weights is None:
        return ans
    else:
        assert len(weights) == len(y_pred)

    return ans * weights

# --------------------------
# Satisfy catboost interface
# --------------------------

# class CensoredMeanSquaredErrorObjective(object):
#     def calc_ders_range(self, y_pred, y_true, weights):
#         assert len(y_pred) == len(y_true)
#         if weights is not None:
#             assert len(weights) == len(y_pred)
#             weights = np.asarray(weights)
#
#         grad = gradient(np.asarray(y_pred), np.asarray(y_true), weights)
#         hess = hessian(np.asarray(y_pred), np.asarray(y_true), weights)
#
#         return list(zip(-grad, -hess))
#
# class CensoredMeanSquaredErrorMetric(object):
#     def is_max_optimal(self):
#         return False  # чем меньше ошибка, тем лучше
#
#     def evaluate(self, approxes, target, weight):
#         y_pred = np.asarray(approxes[0])
#         y_true = np.asarray(target)
#         return evaluate(y_pred, y_true, weight)
#
#     def get_final_error(self, error, weight):
#         return error / weight

class CensoredMeanSquaredErrorObjective(object):
    def calc_ders_range(self, y_pred, y_true, weights):
        assert len(y_pred) == len(y_true)
        if weights is not None:
            assert len(weights) == len(y_pred)

        result = []
        for index in range(len(y_true)):
            pred = y_pred[index]
            true = y_true[index]

            if true != 60:
                der1 = 2 * (pred - true)
                der2 = 2.0
            else:
                if pred >= true:
                    der1 = 0.0
                    der2 = 0.0
                else:
                    der1 = 2 * (pred - true)
                    der2 = 2.0

            if weights is not None:
                der1 *= weights[index]
                der2 *= weights[index]

            result.append((-der1, -der2))
        return result


class CensoredMeanSquaredErrorMetric(object):
    # Duplicated on purpose — numba can't JIT-compile calls to external functions
    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, target, weight):
        y_pred = approxes[0]
        y_true = target

        error_sum = 0.0
        weight_sum = 0.0
        for index in range(len(y_true)):
            pred = y_pred[index]
            true = y_true[index]

            if true != 60:
                err = (pred - true) ** 2
            else:
                err = 0.0 if pred >= true else (pred - true) ** 2

            w = weight[index] if weight is not None else 1.0
            error_sum += err * w
            weight_sum += w

        return error_sum, weight_sum

    def get_final_error(self, error, weight):
        return error / weight

# --------------------------
# Satisfy xgboost interface
# --------------------------

def xgb_gradient(y_pred: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
    '''Compute the gradient for CMSE.'''
    y_true = dtrain.get_label()
    weights = dtrain.get_weight()

    if len(weights) == 0:
        weights = None

    return gradient(y_pred, y_true, weights)

def xgb_hessian(y_pred: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
    '''Compute the hessian for CMSE.'''
    y_true = dtrain.get_label()
    weights = dtrain.get_weight()

    if len(weights) == 0:
        weights = None

    return hessian(y_pred, y_true, weights)     ### reg_lambda > 0

# def xgb_cmse_objective(predt: np.ndarray,
#                 dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
#     '''Censored Mean Squared Error objective'''
#     grad = xgb_gradient(predt, dtrain)
#     hess = xgb_hessian(predt, dtrain)
#     return grad, hess

def xgb_cmse_objective(y_true: np.ndarray, y_pred: np.ndarray, *, sample_weight: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    '''Censored Mean Squared Error objective'''
    grad = gradient(y_pred, y_true, sample_weight)
    hess = hessian(y_pred, y_true, sample_weight)
    return grad, hess

# def xgb_cmse_metric(y_pred: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
#     '''Censored Mean Squared Error metric'''
#     y_true = dtrain.get_label()
#     weights = dtrain.get_weight()
#
#     if len(weights) == 0:
#         weights = None
#
#     er, ws = evaluate(y_pred, y_true, weights)
#     return 'CMSE', er / ws

def xgb_cmse_metric(y_true: np.ndarray, y_pred: np.ndarray, *, sample_weight: Optional[np.ndarray] = None) -> float:
    '''Censored Mean Squared Error metric (sklearn API)'''
    er, ws = evaluate(y_pred, y_true, sample_weight)
    return er / ws