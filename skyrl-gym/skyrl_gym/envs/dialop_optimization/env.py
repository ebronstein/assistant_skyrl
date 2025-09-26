from typing import Dict, Any, Optional, List, Tuple

import numpy as np
from openai import OpenAI

from skyrl_gym.envs.base_text_env import (
    BaseTextEnv,
    BaseTextEnvStepOutput,
    MessageType,
    ConversationType,
)
from .output_parser import DialOpOptimizationOutputParser
from .types import AgentType, DialOpOptimizationActionType, HistoryElement
from .dataset import Table
from .prompt_manager import make_prompt


def get_last_valid_action_type(
    history: List[HistoryElement],
) -> Optional[DialOpOptimizationActionType]:
    """Get the action type of the last valid action.

    Args:
        history: The history of actions taken so far.

    Returns:
        The action type of the last valid action or None if there is no valid action.
    """
    if not history:
        return None

    last_action_type = history[-1]["action_type"]
    if last_action_type != "error":
        return last_action_type

    # If the last action type is "error", that means it is feedback from the previous
    # invalid action. We need to find the last valid action.
    assert len(history) > 1, "An error action should not be the first action."
    last_action_type = None
    for idx in range(len(history) - 3, -1, -2):
        elem = history[idx]
        if elem["action_type"] != "error":
            last_action_type = elem["action_type"]
            break

    return last_action_type


def is_action_allowed(
    action_type: DialOpOptimizationActionType,
    history: List[HistoryElement],
) -> Optional[str]:
    """Check if the action is allowed.

    Args:
        action_type: The type of action that is being taken.
        history: The history of actions taken so far.

    Returns:
        An error message if the action is not allowed, None otherwise.
    """
    last_action_type = get_last_valid_action_type(history)

    if last_action_type in [None, "chat", "reject"]:
        if action_type == "chat" or action_type == "propose_solution":
            return None
        else:
            return "The only allowed actions at this time are '[chat]' and '[propose_solution]'."

    if last_action_type == "propose_solution":
        if action_type == "accept" or action_type == "reject":
            return None
        else:
            return "The only allowed actions after your partner proposed a solution are '[accept]' and '[reject]'."

    if last_action_type == "accept":
        return "No actions are allowed after a solution has been accepted. The conversation is over."

    return f"Invalid action type: {action_type}. Valid actions are: chat, propose_solution, accept, and reject."


class DialOpOptimizationEnv(BaseTextEnv):
    """
    Environment for DialOp Optimization.
    """

    def __init__(self, env_config: Dict[str, Any], extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert (
            "ground_truth" in extras["reward_spec"]
        ), "ground_truth is required in reward_spec field"

        self.ground_truth = extras["reward_spec"]["ground_truth"]
        self.max_reward = extras["extra_info"]["max_reward"]
        self.output_parser = DialOpOptimizationOutputParser(
            len(extras["extra_info"]["table_values"])
        )
        self.history: List[HistoryElement] = []

        # Initialize the table. Format the values as a 2D numpy array because Parquet
        # doesn't support nested arrays.
        values = extras["extra_info"]["table_values"]
        self.table = Table(values=np.array([row for row in values]))
        self.user_table = extras["extra_info"]["user_table"]
        self.assistant_table = extras["extra_info"]["assistant_table"]

        # Which agent's turn it is to act
        self._agent_selection: AgentType = "user"

        self.user_client = OpenAI(**env_config["openai_client"])
        self.model = env_config["model"]

    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        """
        Return the first prompt to be given to the model and optional metadata.
        """
        assert len(prompt) == 1, "The prompt must contain only one message"
        assert prompt[0]["role"] == "system", "Expected system prompt"
        self.system_prompt = prompt[0]["content"]

        assert self._agent_selection == "user"
        self._generate_user_actions()

        return (
            make_prompt(
                self._agent_selection,
                self.history,
                self.system_prompt,
                self.user_table,
                self.assistant_table,
            ),
            {},
        )

    def step(self, action: str) -> BaseTextEnvStepOutput:
        assert self._agent_selection == "assistant"
        assistant_step_output = self._step_agent(self._agent_selection, action)

        if self._agent_selection == "assistant":
            return assistant_step_output

        user_outputs = self._generate_user_actions()
        observations = sum((output["observations"] for output in user_outputs), [])

        return BaseTextEnvStepOutput(
            observations=observations,
            reward=user_outputs[-1]["reward"],
            done=user_outputs[-1]["done"],
        )

    def _generate_user_actions(
        self, max_actions: int = 5
    ) -> List[BaseTextEnvStepOutput]:
        """Generate user actions."""
        observations = []
        while self._agent_selection == "user" and len(observations) < max_actions:
            user_response = self.user_client.responses.create(
                model=self.model,
                input=make_prompt(
                    self._agent_selection,
                    self.history,
                    self.system_prompt,
                    self.user_table,
                    self.assistant_table,
                ),
            )
            user_action = user_response.output[0].content[0].text

            user_step_output = self._step_agent(self._agent_selection, user_action)
            observations.append(user_step_output)
            if user_step_output["done"]:
                break

        return observations

    def _step_agent(self, agent: AgentType, action: str) -> BaseTextEnvStepOutput:
        """Step the agent and update the history.

        Args:
            agent: The agent that is taking the action.
            action: The action that the agent is taking.

        Returns:
            The output of the agent's action.
        """
        self.turns += 1

        action_type, action_content, error_message = self.output_parser.parse_action(
            action
        )

        # If the action was parsed successfully, check if the action is allowed
        if error_message is None:
            error_message = is_action_allowed(action_type, self.history)

        # Add the action to the history
        self.history.append(
            {
                "agent": agent,
                "action_type": action_type,
                "action_content": action_content,
                "message": action,
                "error_message": error_message,
            }
        )

        # If it is the user's turn, the user's actions are observed by the assistant.
        # Otherwise, no observations are returned because the environment does not
        # produce any observations from the assistant's perspective.
        if agent == "user":
            observations = [{"role": agent, "content": action}]
        else:
            observations = []

        # If the action is invalid, add the error message as an observation
        if error_message is not None:
            # The feedback role is the opposite of the agent's role
            feedback_role = "user" if agent == "assistant" else "assistant"
            self.history.append(
                {
                    "agent": feedback_role,
                    "action_type": "error",
                    "action_content": error_message,
                    "message": error_message,
                    "error_message": None,
                }
            )
            observations.append({"role": feedback_role, "content": error_message})
            return BaseTextEnvStepOutput(
                observations=observations, reward=0.0, done=False
            )

        # Update the agent selection. The next agent to act is the opposite of the
        # current agent if the action is a chat or propose_solution.
        # Otherwise (accept or reject) it is still the current agent's turn.
        if action_type in ["chat", "propose_solution"]:
            self._agent_selection = "assistant" if agent == "user" else "user"

        done = action_type == "accept"
        reward = self._compute_reward() if done else 0.0

        return BaseTextEnvStepOutput(
            observations=observations, reward=reward, done=done
        )

    def _compute_reward(self) -> float:
        """Compute the reward for the last solution proposed by the assistant."""
        last_solution = None
        for elem in reversed(self.history):
            if elem["action_type"] == "propose_solution":
                last_solution = elem["action_content"]
                break

        if last_solution is None:
            return 0.0

        return self.table.score(last_solution) / self.max_reward
