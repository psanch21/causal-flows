from .mlp import MLP

from causal_nf.modules import module_dict, module_params_dict

module_dict_mlp = {}
module_dict_mlp["mlp"] = MLP

module_dict.update(module_dict_mlp)
for module_name, module_class in module_dict_mlp.items():
    module_params_dict[module_name] = module_class.kwargs
