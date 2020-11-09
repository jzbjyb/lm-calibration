import gin
from .test import build_test
from .unifiedqa import build_uq
from .utils import read_score_data, convert_data_to_dmatrix


@gin.configurable
def build(neg_method: str='indicator', ret_ind: int=0, ret_method: str='q-prepend'):
  build_test(neg_method=neg_method, ret_ind=ret_ind, ret_method=ret_method)
  build_uq(neg_method=neg_method, ret_ind=ret_ind, ret_method=ret_method)
