import gin
from .test import build_test
from .unifiedqa import build_uq


@gin.configurable
def build(neg_method: str='indicator'):
  build_test(neg_method=neg_method)
  build_uq(neg_method=neg_method)
