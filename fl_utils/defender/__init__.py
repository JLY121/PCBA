# 导入所有聚合方法
from .avg import average_shrink_models
from .fedprox import fedprox_aggregate
from .fednova import fednova_aggregate_global_model
from .clip import clip_aggregate
from .krum import krum
from .multkrum import multi_krum
from .dp import dp_avg
from .median import median_shrink_models
from .rfa import geometric_median_update
from .rlr import robust_lr_aggregate
from .crfl import crfl_agg
from .bulyan import bulyan_aggregate_global_model
from .deep import deepsight_aggregate_global_model_v2
from .foolsgold import foolsgold_aggregate
from .flame import flame_aggregate
from .feddmc import feddmc_aggregate
from .indicator import indicator_defense
from .alignIns import alignins_aggregate

__all__ = [
    'average_shrink_models',
    'fedprox_aggregate',
    'fednova_aggregate_global_model',
    'clip_aggregate',
    'krum',
    'multi_krum',
    'dp_avg',
    'median_shrink_models',
    'geometric_median_update',
    'robust_lr_aggregate',
    'crfl_agg',
    'bulyan_aggregate_global_model',
    'deepsight_aggregate_global_model_v2',
    'foolsgold_aggregate',
    'flame_aggregate',
    'feddmc_aggregate',
    'indicator_defense',
    'alignins_aggregate'
] 