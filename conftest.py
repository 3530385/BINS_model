import pytest
from model_bins import BINS

@pytest.fixture
def bins_with_random():
    bins = BINS(psi=90, theta=3, gamma=0,
                dpsi=-0.5, dwbx=1, dwby=0.5,
                dabx=1, daby=0.5, sigma_a=1,
                Tka=0.2, sigma_w=2, Tkw=0.1,
                rand=True)
    return bins


@pytest.fixture
def bins_without_random():
    bins = BINS(psi=90, theta=3, gamma=0,
                dpsi=-0.5, dwbx=1, dwby=0.5,
                dabx=1, daby=0.5, sigma_a=1,
                Tka=0.2, sigma_w=2, Tkw=0.1,
                rand=False)
    return bins
