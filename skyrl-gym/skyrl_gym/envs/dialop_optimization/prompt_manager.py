from typing import List

from skyrl_gym.envs.base_text_env import ConversationType

from .types import AgentType, HistoryElement


def make_prompt(
    agent: AgentType,
    history: List[HistoryElement],
    system_prompt: str,
    user_table: str,
    assistant_table: str,
) -> str:
    """Make a prompt for the given agent and history."""
    assert agent in ["user", "assistant"], f"Invalid agent: {agent}"

    table = user_table if agent == "user" else assistant_table
    system_prompt += (
        f"\nHere is your table of reviewer-paper similarity scores:\n{table}"
    )
    prompt = [{"content": system_prompt, "role": "system"}]

    if agent == "user":
        prompt.append(
            {
                "role": "user",
                "content": "Hello, let’s assign these papers to the reviewers.",
            }
        )

    # Reverse the roles if the agent is the user because the LLM expects to play the
    # "assistant" role.
    history_prompt = make_history_prompt(history, reverse_roles=agent == "user")
    prompt.extend(history_prompt)
    return prompt


def make_history_prompt(
    history: List[HistoryElement], reverse_roles: bool = False
) -> ConversationType:
    """Make a prompt for the history."""
    prompt = []
    for elem in history:
        if reverse_roles:
            role = "assistant" if elem["agent"] == "user" else "user"
        else:
            role = elem["agent"]
        prompt.append({"role": role, "content": elem["message"]})
    return prompt
