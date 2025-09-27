from skyrl_gym.envs.dialop_optimization.env import get_last_valid_action_type
from .utils import make_placeholder_error_element


def test_empty_history():
    assert get_last_valid_action_type([]) is None


def test_first_action_is_invalid():
    history = [
        {
            "agent": "user",
            "action_type": "reject",
            "action_content": None,
            "message": "[reject]",
            "error_message": None,
        },
        make_placeholder_error_element("assistant"),
    ]

    assert get_last_valid_action_type(history) is None


def test_last_user_action_is_invalid():
    history = [
        {
            "agent": "assistant",
            "action_type": "propose_solution",
            "action_content": "foo",
            "message": "[propose_solution] foo",
            "error_message": None,
        },
        {
            "agent": "user",
            "action_type": "chat",
            "action_content": "Good idea!",
            "message": "Good idea!",
            "error_message": "An error message.",
        },
        make_placeholder_error_element("assistant"),
    ]
    assert get_last_valid_action_type(history) is "propose_solution"


def test_last_assistant_action_is_invalid():
    history = [
        {
            "agent": "user",
            "action_type": "propose_solution",
            "action_content": "foo",
            "message": "[propose_solution] foo",
            "error_message": None,
        },
        {
            "agent": "assistant",
            "action_type": "chat",
            "action_content": "Good idea!",
            "message": "Good idea!",
            "error_message": "An error message.",
        },
        make_placeholder_error_element("user"),
    ]
    assert get_last_valid_action_type(history) is "propose_solution"


def test_multiple_invalid_actions():
    history = [
        {
            "agent": "user",
            "action_type": "chat",
            "action_content": "hi",
            "message": "hi",
            "error_message": None,
        },
        {
            "agent": "assistant",
            "action_type": "chat",
            "action_content": "hello",
            "message": "hello",
            "error_message": None,
        },
        {
            "agent": "user",
            "action_type": "reject",
            "action_content": None,
            "message": "[reject]",
            "error_message": None,
        },
        make_placeholder_error_element("assistant"),
        {
            "agent": "user",
            "action_type": "reject",
            "action_content": None,
            "message": "[reject]",
            "error_message": None,
        },
        make_placeholder_error_element("assistant"),
        {
            "agent": "user",
            "action_type": "reject",
            "action_content": None,
            "message": "[reject]",
            "error_message": None,
        },
        make_placeholder_error_element("assistant"),
    ]
    assert get_last_valid_action_type(history) is "chat"


def test_valid_action():
    history = [
        {
            "agent": "user",
            "action_type": "propose_solution",
            "action_content": "foo",
            "message": "[propose_solution] foo",
            "error_message": None,
        },
        {
            "agent": "assistant",
            "action_type": "accept",
            "action_content": None,
            "message": "[accept]",
            "error_message": None,
        },
    ]
    assert get_last_valid_action_type(history) is "accept"
