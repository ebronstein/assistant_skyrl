from unittest.mock import MagicMock

import numpy as np
import pytest

from skyrl_gym.envs.dialop_optimization.dataset import Table
from skyrl_gym.envs.dialop_optimization.env import compute_reward
from .utils import make_history_element, make_placeholder_error_element


@pytest.mark.parametrize(
    "history",
    [
        # No solution proposed yet
        [make_history_element("user", "chat", "Hello")],
        # Solution not accepted yet
        [make_history_element("user", "propose_solution", "solution")],
        # Solution rejected
        [
            make_history_element("user", "propose_solution", "solution"),
            make_history_element("assistant", "reject", None),
        ],
        # Invalid accept action
        [
            make_history_element("user", "chat", "Hello"),
            make_history_element("assistant", "accept", None),
            make_placeholder_error_element("user"),
        ],
    ],
)
def test_no_accepted_solution(history):
    assert compute_reward(history, MagicMock(), 1.0) == 0.0


def test_errors_if_accept_without_solution():
    history = [
        make_history_element("user", "chat", "Hello"),
        make_history_element("assistant", "accept", None),
    ]
    with pytest.raises(AssertionError):
        compute_reward(history, MagicMock(), 1.0)


def test_max_reward():
    """Test that the reward is 1.0 if the best assignment is proposed and accepted."""
    table = Table(num_rows=8, num_cols=8)
    best_assignment, best_assignment_reward = table.find_max_value_known_assignment(
        [np.ones((8, 8)), np.ones((8, 8))]
    )
    history = [
        make_history_element("user", "propose_solution", best_assignment),
        make_history_element("assistant", "accept", None),
    ]

    assert compute_reward(history, table, best_assignment_reward) == 1.0
