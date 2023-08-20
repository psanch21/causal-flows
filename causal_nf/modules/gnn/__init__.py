from .gcn import GCN
from .gat import GAT
from .gin import GIN
from .gine import GINE
from .pna import PNA
from .disjoint_pna import DisjointPNA
from causal_nf.modules import module_dict, module_params_dict

module_dict_gnn = {}
module_dict_gnn["gcn"] = GCN
module_dict_gnn["gat"] = GAT
module_dict_gnn["gin"] = GIN
module_dict_gnn["gine"] = GINE
module_dict_gnn["pna"] = PNA
module_dict_gnn["disjoint_pna"] = DisjointPNA

module_dict.update(module_dict_gnn)
for module_name, module_class in module_dict_gnn.items():
    module_params_dict[module_name] = module_class.kwargs

pooling_dict = {}
