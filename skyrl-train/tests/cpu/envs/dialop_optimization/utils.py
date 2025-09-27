def make_history_element(
    agent, action_type, action_content, message=None, error_message=None
):
    if message is None:
        if action_type == "chat":
            message = action_content
        elif action_type == "propose_solution":
            message = f"[propose_solution] {action_content}"
        else:
            message = f"[{action_type}]"

    return {
        "agent": agent,
        "action_type": action_type,
        "action_content": action_content,
        "message": message,
        "error_message": error_message,
    }


def make_placeholder_error_element(agent):
    return {
        "agent": agent,
        "action_type": "error",
        "action_content": "An error message.",
        "message": "[error]",
        "error_message": "An error message.",
    }
