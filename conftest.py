import pytest
from model_bins import BINS


@pytest.fixture
def bins_with_random():
    bins = BINS(psi=90, theta=-3, gamma=2,
                dpsi=1, dwbx=-2, dwby=1,
                # dpsi=1, dwbx=0, dwby=0,
                dabx=-2, daby=1, sigma_a=0.5,
                # dabx=0, daby=0, sigma_a=0.5,
                Tka=0.2, sigma_w=.05, Tkw=0.1,
                rand=True)
    return bins


@pytest.fixture
def bins_without_random():
    bins = BINS(psi=90, theta=-3, gamma=2,
                dpsi=1, dwbx=-2, dwby=1,
                # dpsi=1, dwbx=0, dwby=0,
                dabx=-2, daby=1, sigma_a=0.5,
                # dabx=0, daby=0, sigma_a=0.5,
                Tka=0.2, sigma_w=.05, Tkw=0.1,
                rand=False)
    return bins
