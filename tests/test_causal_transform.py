import pytest

from causal_nf.sem_equations import sem_dict
from causal_nf.transforms import CausalEquations


@pytest.mark.parametrize("name", ["chain", "triangle"])
@pytest.mark.parametrize("sem_name", ["non-linear"])
def test_causal_equations(name, sem_name):
    sem = sem_dict[name](sem_name=sem_name)

    transform = CausalEquations(
        functions=sem.functions, inverses=sem.inverses, derivatives=None
    )
