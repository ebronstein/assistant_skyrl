import pytest

from skyrl_gym.envs.dialop_optimization.output_parser import (
    DialOpOptimizationOutputParser,
)


@pytest.fixture
def output_parser():
    return DialOpOptimizationOutputParser(num_papers=8)


def test_chat(output_parser):
    action_type, action_content, error_message = output_parser.parse_action("Hello")
    assert action_type == "chat"
    assert action_content == "Hello"
    assert error_message is None


def test_accept(output_parser):
    action_type, action_content, error_message = output_parser.parse_action("[accept]")
    assert action_type == "accept"
    assert action_content is None
    assert error_message is None


def test_accept_with_content(output_parser):
    message = "[accept] hi"
    action_type, action_content, error_message = output_parser.parse_action(message)
    assert action_type is None
    assert action_content is None
    assert "accept action should not have content" in error_message


def test_reject(output_parser):
    action_type, action_content, error_message = output_parser.parse_action("[reject]")
    assert action_type == "reject"
    assert action_content is None
    assert error_message is None


def test_reject_with_content(output_parser):
    message = "[reject] hi"
    action_type, action_content, error_message = output_parser.parse_action(message)
    assert action_type is None
    assert action_content is None
    assert "reject action should not have content" in error_message


def test_nonexistent_action(output_parser):
    message = "[nonexistent] hi"
    action_type, action_content, error_message = output_parser.parse_action(message)
    assert action_type is None
    assert action_content is None
    assert "Invalid action type: nonexistent" in error_message


def test_empty_assignment(output_parser):
    action_type, action_content, error_message = output_parser.parse_action(
        "[propose_solution]"
    )
    assert action_type == "propose_solution"
    assert action_content is None
    assert "Empty assignment" in error_message


def test_assignment_with_wrong_number_of_assignments(output_parser):
    action_type, action_content, error_message = output_parser.parse_action(
        "[propose_solution] Ava Li: swag, Daniel Nguyen: bleu"
    )
    assert action_type == "propose_solution"
    assert action_content is None
    assert "Expected 8 assignments, but got 2" in error_message
