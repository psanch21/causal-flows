from .minmax import MinMaxScaler_n11, MinMaxScaler_01, MinMaxScaler
from .min import MinScaler
from .standard import StandardScaler
from .scale import ScaleScaler
from .scalediff import ScaleDiffScaler
from .identity import IdentityScaler
from .heterogeneous import HeterogeneousScaler
from .heterogeneous_object import HeterogeneousObjectScaler

scalers_dict = {
    "minn1_max1": MinMaxScaler_n11,
    "min0_max1": MinMaxScaler_01,
    "min0": MinScaler,
    "std": StandardScaler,
    "scale": ScaleScaler,
    "scale_diff": ScaleDiffScaler,
    "identity": IdentityScaler,
}
