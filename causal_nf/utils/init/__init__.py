from .normal import init_normal
from .xavier import init_xavier
from .none import init_none


def get_init_fn(cfg_model):
    init = cfg_model.init
    activation = cfg_model.act
    if init is None:
        return init_none()
    if "xavier" in init:
        act_name = activation.split("__")[0]
        if act_name == "lrelu":
            act_name = "leaky_relu"

        if act_name in ["leaky_relu"]:
            return init_xavier(
                act_name, param=float(activation.split("__")[1].replace("_", "."))
            )
        else:
            return init_xavier(act_name)
    elif "normal" in init:
        std = float(init.split("__")[1].replace("_", "."))
        return init_normal(std)
    else:
        raise NotImplementedError
