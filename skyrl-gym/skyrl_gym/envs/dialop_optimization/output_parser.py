from typing import Optional, get_args, Tuple, List
import re
import numpy as np
from numpy.typing import NDArray

from .dataset import DialOpOptimizationSolution
from .types import DialOpOptimizationActionType


TASKS = [
    "BLEU: a Method for Automatic Evaluation of MT",
    "Electra: Pre-training Text Encoders as Discriminators",
    "GloVe: Global Vectors for Word Representation",
    "GLUE: A Multi-Task Benchmark and Analysis Platform for NLU",
    "LLaMA: Open and Efficient Foundation Language Models",
    "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
    "QuAC: Question Answering in Context",
    "SWAG: An Adversarial Dataset for Commonsense Inference",
]

WORKERS = [
    "Ava Li",
    "Daniel Nguyen",
    "Sofia Patel",
    "Andrei Petrov",
    "Morgan Reed",
    "Joseph Santos",
    "Ethan Smith",
    "Noah Wilson",
]


def parse_assignment(
    assignment_string: str, num_papers: int
) -> Tuple[Optional[DialOpOptimizationSolution], Optional[str]]:
    """
    Parse a string assignment of reviewers to papers.

    Args:
        assignment_string: Comma-separated list of "paper: reviewer"
            (i.e., "task: worker") items
        num_papers: Expected number of assignments

    Returns:
        A tuple (solution, error_message):
        - solution is a Numpy array whereresult[i] = j means reviewer i is
          assigned paper j
        - error_message is an error message if parsing fails. If parsing is
          successful, the error message is None.

    Raises:
        ValueError: num_papers is not positive
    """
    if num_papers <= 0:
        raise ValueError(f"num_papers must be positive, but got {num_papers}")

    if not assignment_string:
        return None, "Empty assignment"

    # Split by comma and strip whitespace
    assignments = [item.strip() for item in assignment_string.split(",")]

    # Check number of assignments
    if len(assignments) != num_papers:
        return None, (f"Expected {num_papers} assignments, but got {len(assignments)}")

    # Initialize result array: result[reviewer_idx] = paper_idx
    result = np.full(len(WORKERS), -1, dtype=np.int8)
    used_tasks = set()
    used_workers = set()

    for assignment in assignments:
        # Check if assignment contains colon
        if ":" not in assignment:
            return None, f"Misformatted assignment (missing colon): '{assignment}'"

        # Split by colon - split on last colon to handle task names with colons
        colon_idx = assignment.rfind(":")
        if colon_idx == -1:
            return None, f"Misformatted assignment (missing colon): '{assignment}'"

        task_str = assignment[:colon_idx].strip()
        worker_str = assignment[colon_idx + 1 :].strip()

        if not task_str or not worker_str:
            return None, (
                f"Misformatted assignment (empty task or worker): '{assignment}'"
            )

        # Match task
        task_idx, error_message = match_task(task_str, TASKS)
        if error_message:
            return None, error_message

        # Match worker
        worker_idx, error_message = match_worker(worker_str, WORKERS)
        if error_message:
            return None, error_message

        # Check for duplicates
        if task_idx in used_tasks:
            task_name = TASKS[task_idx].split(":")[0]
            return None, f"Task '{task_name}' is assigned multiple times"

        if worker_idx in used_workers:
            worker_name = WORKERS[worker_idx]
            return None, f"Worker '{worker_name}' is assigned multiple times"

        used_tasks.add(task_idx)
        used_workers.add(worker_idx)

        # Assign paper to reviewer: result[reviewer_idx] = paper_idx
        result[worker_idx] = task_idx

    return result, None


def match_task(task_str: str, tasks: List[str]) -> Tuple[Optional[int], Optional[str]]:
    """Match task string to tasks list using word overlap.

    Args:
        task_str: The task string to match.
        tasks: The list of tasks to match against.

    Returns:
        The index of the matched task and an error message. If matching is successful,
        the error message is None. Otherwise, the index is None.
    """
    task_words = set(word.lower() for word in task_str.split())

    best_matches = []
    max_overlap = 0

    for i, task in enumerate(tasks):
        # Get all words from the task (including short name)
        task_all_words = set(word.lower() for word in task.replace(":", " ").split())

        overlap = len(task_words.intersection(task_all_words))

        if overlap > max_overlap:
            max_overlap = overlap
            best_matches = [i]
        elif overlap == max_overlap and overlap > 0:
            best_matches.append(i)

    if max_overlap == 0:
        idx = None
        error_message = f"No matching task found for '{task_str}'."
    elif len(best_matches) > 1:
        idx = None
        error_message = (
            f"'{task_str}' matches multiple tasks: {task_names}."
            "Unclear which task to assign."
        )
    else:
        idx = best_matches[0]
        error_message = None

    return idx, error_message


def match_worker(
    worker_str: str, workers: List[str]
) -> Tuple[Optional[int], Optional[str]]:
    """Match worker string to workers list using exact name matching.

    Args:
        worker_str: The worker string to match.
        workers: The list of workers to match against.

    Returns:
        The index of the matched worker and an error message. If matching is successful,
        the error message is None. Otherwise, the index is None.
    """
    worker_lower = worker_str.lower().strip()

    matching_worker_idxs = []
    for i, worker in enumerate(workers):
        worker_parts = worker.split()
        first_name = worker_parts[0].lower()
        last_name = worker_parts[1].lower() if len(worker_parts) > 1 else ""
        full_name = worker.lower()

        if (
            worker_lower in [first_name, last_name, full_name]
            or first_name in worker_lower
            or last_name in worker_lower
            or full_name in worker_lower
        ):
            matching_worker_idxs.append(i)

    if not matching_worker_idxs:
        idx = None
        error_message = f"No reviewers match '{worker_str}'."
    elif len(matching_worker_idxs) > 1:
        idx = None
        error_message = (
            f"Multiple reviewers found matching '{worker_str}'. "
            "Unclear which reviewer to assign."
        )
    else:
        idx = matching_worker_idxs[0]
        error_message = None

    return idx, error_message


class DialOpOptimizationOutputParser:
    """Output parser for the DialOp optimization environment."""

    def __init__(
        self,
        num_papers: int,
    ):
        """Initialize the output parser.

        Args:
            num_papers: Number of papers
        """
        self.num_papers = num_papers

    def parse_action(
        self, output: str
    ) -> Tuple[Optional[DialOpOptimizationActionType], Optional[str], Optional[str]]:
        """Parse output.

        This method expects the output to be in the following format:
        [action_name] content

        Examples:
        - [chat] Hi!
        - [proposal] BLEU: Noah Wilson, Electra: Daniel Nguyen
        - [accept]
        - [reject]

        Args:
            output: The raw output string.
            agent: The agent type that generated the output.

        Returns:
            A tuple of the action type, action content, and an error message if
            parsing fails. If parsing is successful, the error message is None.
            Otherwise, the action type and content are None.
        """
        # Match pattern: [action_name] optional_content
        match = re.match(r"^\[(\w+)\](?:\s+(.*))?$", output.strip(), re.DOTALL)

        # If no action is specified, it is a chat action.
        if not match:
            return "chat", output, None

        action_type = match.group(1).lower()
        content = match.group(2) if match.group(2) else ""

        # Validate content based on action type
        if action_type == "chat":
            return action_type, content, None
        elif action_type == "propose_solution":
            assignment, error_message = parse_assignment(content, self.num_papers)
            return action_type, assignment, error_message
        elif action_type == "accept" or action_type == "reject":
            if content.strip():
                return (
                    None,
                    None,
                    f"{action_type} actions should not have content. Simply return '[{action_type}]'.",
                )
            return action_type, None, None
        else:
            return (
                None,
                None,
                f"Invalid action type: {action_type}. Valid actions are: {get_args(DialOpOptimizationActionType)}",
            )
