import pytest

from skyrl_gym.envs.dialop_optimization.env import is_action_allowed
from .utils import make_history_element, make_placeholder_error_element


@pytest.mark.parametrize(
    "action_type, is_allowed",
    [
        ("chat", True),
        ("propose_solution", True),
        ("accept", False),
        ("reject", False),
    ],
)
def test_empty_history(action_type, is_allowed):
    err_msg = is_action_allowed(action_type, [])
    assert (err_msg is None) == is_allowed


@pytest.mark.parametrize(
    "action_type, is_allowed",
    [
        ("chat", True),
        ("propose_solution", True),
        ("accept", False),
        ("reject", False),
    ],
)
def test_after_chat_action(action_type, is_allowed):
    history = [
        make_history_element("user", "chat", "Hello"),
    ]
    err_msg = is_action_allowed(action_type, history)
    assert (err_msg is None) == is_allowed


@pytest.mark.parametrize(
    "action_type, is_allowed",
    [
        ("chat", True),
        ("propose_solution", True),
        ("accept", False),
        ("reject", False),
    ],
)
def test_after_reject_action(action_type, is_allowed):
    history = [
        make_history_element("user", "propose_solution", "foo"),
        make_history_element("assistant", "reject", None),
    ]
    err_msg = is_action_allowed(action_type, history)
    assert (err_msg is None) == is_allowed


@pytest.mark.parametrize(
    "action_type, is_allowed",
    [
        ("chat", False),
        ("propose_solution", False),
        ("accept", True),
        ("reject", True),
    ],
)
def test_after_propose_solution_action(action_type, is_allowed):
    history = [
        make_history_element("user", "propose_solution", "foo"),
    ]
    err_msg = is_action_allowed(action_type, history)
    assert (err_msg is None) == is_allowed


@pytest.mark.parametrize(
    "action_type, is_allowed",
    [
        ("chat", False),
        ("propose_solution", False),
        ("accept", False),
        ("reject", False),
    ],
)
def test_after_accept_action(action_type, is_allowed):
    history = [
        make_history_element("user", "propose_solution", "foo"),
        make_history_element("assistant", "accept", None),
    ]
    err_msg = is_action_allowed(action_type, history)
    assert (err_msg is None) == is_allowed


def test_after_reject_action_chat_allowed():
    history = [
        make_history_element(
            "assistant", "propose_solution", "solution", "[propose_solution] solution"
        ),
        make_history_element("user", "reject", None, "[reject]"),
    ]
    assert is_action_allowed("chat", history) is None


@pytest.mark.parametrize(
    "action_type, is_allowed",
    [
        ("chat", False),
        ("propose_solution", False),
        ("accept", True),
        ("reject", True),
    ],
)
def test_last_action_error(action_type, is_allowed):
    history = [
        make_history_element("user", "propose_solution", "foo"),
        make_history_element("assistant", "chat", "Hello"),
        make_placeholder_error_element("user"),
    ]
    err_msg = is_action_allowed(action_type, history)
    assert (err_msg is None) == is_allowed


def test_invalid_action_type():
    assert "Invalid action type" in is_action_allowed("invalid", [])
