import pytest

from skyrl_gym.envs.dialop_optimization.prompt_manager import (
    make_history_prompt,
    make_prompt,
)


from .utils import make_history_element, make_placeholder_error_element


def test_make_empty_history_prompt():
    prompt = make_history_prompt([])
    assert prompt == []


def test_make_history_prompt():
    history = [
        make_history_element("user", "chat", "Hello"),
        make_history_element("assistant", "chat", "Hi"),
    ]
    prompt = make_history_prompt(history)
    expected_prompt = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]

    assert prompt == expected_prompt


def test_make_history_prompt_reverse_roles():
    history = [
        make_history_element("user", "chat", "Hello"),
        make_history_element("assistant", "chat", "Hi"),
    ]
    prompt = make_history_prompt(history, reverse_roles=True)
    expected_prompt = [
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "Hi"},
    ]
    assert prompt == expected_prompt


def test_make_prompt_invalid_agent():
    with pytest.raises(AssertionError):
        make_prompt("invalid", [], "", "", "")


def test_make_prompt_for_assistant():
    history = [
        make_history_element("user", "chat", "Hello"),
        make_history_element("assistant", "chat", "Hi"),
    ]

    system_prompt = "You are a helpful assistant."
    user_table = "USER TABLE"
    assistant_table = "ASSISTANT TABLE"

    prompt = make_prompt(
        "assistant", history, system_prompt, user_table, assistant_table
    )

    expected_prompt_without_system = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]

    assert len(prompt) == 3

    assert prompt[0]["role"] == "system"
    assert system_prompt in prompt[0]["content"]
    assert assistant_table in prompt[0]["content"]
    assert user_table not in prompt[0]["content"]

    assert prompt[1:] == expected_prompt_without_system


def test_make_prompt_for_user():
    history = [
        make_history_element("user", "chat", "Hello"),
        make_history_element("assistant", "chat", "Hi"),
    ]
    system_prompt = "SYSTEM_PROMPT"
    user_table = "USER TABLE"
    assistant_table = "ASSISTANT TABLE"

    prompt = make_prompt("user", history, system_prompt, user_table, assistant_table)
    expected_prompt_without_system = [
        {
            "role": "user",
            "content": "Hello, let's assign these papers to the reviewers.",
        },
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "Hi"},
    ]

    assert len(prompt) == 4

    assert prompt[0]["role"] == "system"
    assert system_prompt in prompt[0]["content"]
    assert assistant_table not in prompt[0]["content"]
    assert user_table in prompt[0]["content"]

    assert prompt[1:] == expected_prompt_without_system
