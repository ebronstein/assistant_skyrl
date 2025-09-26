"""Preprocess the dataset for the 'dialop_optimization' environment in parquet format."""

import argparse
import random
import os
import csv
import io
from typing import List, Optional, Tuple, TypedDict, Union

import numpy as np
from numpy.typing import NDArray
import scipy.optimize

from datasets import Dataset

# List of reviewer-paper assignments. The value of the i-th element is the paper index
# for the i-th reviewer.
DialOpOptimizationSolution = NDArray[np.int8]


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

TASKS_SHORT = [task.split(":")[0] for task in TASKS]

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

SYSTEM_PROMPT = """You and your partner are area chairs for a conference, and you have to assign reviewers to papers. Each of you has some information about which reviewers would be good for which papers, but you'll have to communicate in order to make the best assignments.

You will see a table of reviewer-paper similarity scores. The higher the score, the better the fit. Your partner has their own table, which may be different from yours. Your goal is to propose a one-to-one matching between reviewers and papers with the highest sum of scores. Note that the scores only show relative fit (e.g., reviewer1 is a better fit than reviewer2), so you cannot and should not compare your scores with your partner's.

You can send messages to your partner, propose assignments, and accept or reject your partner's proposals. Only certain actions are available at certain times:
- At the start of the conversation, you can send a message, or propose an assignment.
- If your partner sent a message, you can send a message or propose an assignment.
- If your partner proposed an assignment, you can only accept or reject it.

If you send a message or propose an assignment, it is your partner's turn (and vice versa). If you reject an assignment, it is still your turn. If you or your partner accept an assignment, the conversation is over.

You have the following actions available:
- Send a message to your partner. This is for discussion purposes, not for formally proposing or rejecting assignments. No formatting is needed. Example: "Hello, how are you?"
- Propose an assignment of reviewers to papers. Format: "[propose_solution] BLEU: a Method for Automatic Evaluation of MT: Sofia Patel, Electra: Pre-training Text Encoders as Discriminators: Ethan Smith, ..." The assignment must be a one-to-one matching between reviewers and papers.
- Accept a proposal from your partner. Format: "[accept]"
- Reject a proposal from your partner. Format: "[reject]"

List of papers:
{TASKS}

List of reviewers:
{WORKERS}
""".format(
    TASKS=TASKS, WORKERS=WORKERS
)

# Maximum number of tables to sample before returning a task
MAX_TABLES = 1000


class Table:
    """Bipartite matching table for reviewer-matching scenario.

    Attributes:
        num_rows: number of rows (reviewers)
        num_cols: number of columns (papers)
        values: a 2D numpy array of costs
        max_val: maximum value for random generation
        empty_val: value to represent unknown/empty cells
    """

    def __init__(
        self,
        num_rows: Optional[int] = None,
        num_cols: Optional[int] = None,
        values: Optional[np.ndarray] = None,
        max_val: int = 100,
        empty_val: int = -1,
    ) -> None:
        """Initialize a Table instance.

        Args:
            num_rows: number of rows (reviewers)
            num_cols: number of columns (papers)
            values: 2D numpy array of costs
            max_val: maximum value for random generation
            empty_val: value to represent unknown/empty cells
        """
        self.max_val = max_val
        self.empty_val = empty_val
        if values is not None:
            assert (
                num_rows is None and num_cols is None
            ), "num_rows and num_cols must be None if values is provided."
            self.values = np.array(values)
            self.num_rows, self.num_cols = self.values.shape
        else:
            assert num_rows is not None and num_cols is not None, (
                "num_rows and num_cols must be defined unless initializing from "
                "values."
            )
            assert (
                num_rows > 0 and num_cols > 0
            ), "num_rows and num_cols must be greater than 0."
            if num_rows != num_cols:
                raise NotImplementedError("Only bipartite matchings are supported.")

            self.num_rows = num_rows
            self.num_cols = num_cols
            self.values = np.random.randint(
                0, self.max_val, (self.num_rows, self.num_cols)
            )

    def set_values(self, values: np.ndarray) -> None:
        """Set the values of the table.

        Args:
            values: 2D numpy array of costs
        """
        self.values = np.copy(values)

    def find_max_value_known_assignment(
        self, knowns: List[np.ndarray]
    ) -> Tuple[DialOpOptimizationSolution, float]:
        """Find the max value assignment given a set of views.

        If the players pool their knowledge, what is the best they can do?

        Args:
            knowns: list of player view mask arrays

        Returns:
            assignment: DialOpOptimizationSolution representing the best assignment.
              assignment[i] = j means reviewer i is assigned paper j
            score: score of the best assignment
        """
        known = np.logical_or(*knowns)
        pooled_expected_table = -self.values * known - self.max_val / 2 * (1 - known)
        rows, cols = scipy.optimize.linear_sum_assignment(pooled_expected_table)
        assert np.array_equal(
            rows, np.arange(self.num_rows)
        ), f"Rows expected to be sorted. Got: {rows}"
        score = self.score(cols)
        return cols, score

    def get_random_view(self, p_cell_observed: float) -> Tuple["Table", np.ndarray]:
        """Return a new Table instance representing a view of this table.

        Args:
            p_cell_observed: probability that each cell is observed

        Returns:
            output_table: new Table instance with masked values
            mask: boolean mask indicating which cells are known
        """
        unknown = np.random.choice(
            2, self.values.shape, p=[p_cell_observed, 1.0 - p_cell_observed]
        )
        output_values: np.ndarray = np.ma.masked_array(self.values, mask=unknown)
        output_values = np.ma.filled(output_values, self.empty_val)
        output_table = Table(values=output_values)
        return output_table, 1 - unknown

    def score(self, assignment: DialOpOptimizationSolution) -> float:
        """Calculate the score of an assignment.

        Args:
            assignment: DialOpOptimizationSolution representing the assignment.
              assignment[i] = j means reviewer i is assigned paper j

        Returns:
            total score of the assignment

        Raises:
            AssertionError: if assignment length doesn't match number of rows
        """
        assert assignment.shape == (
            self.num_rows,
        ), f"{assignment.shape} != ({self.num_rows},)"
        return float(sum([self.values[r][c] for r, c in enumerate(assignment)]))


class DialOpOptimizationTask(TypedDict):
    """DialOp optimization task."""

    table_values: np.ndarray
    solution: DialOpOptimizationSolution
    max_reward: float
    user_table: str
    assistant_table: str
    user_known_mask: np.ndarray
    assistant_known_mask: np.ndarray
    user_scale_factor: int
    assistant_scale_factor: int


class DialOpOptimizationTaskGenerator:
    """DialOp optimization task generator."""

    def __init__(self, num_assignments: int, p_cell_observed: float) -> None:
        self.num_assignments = num_assignments
        self.p_cell_observed = p_cell_observed

    def sample(self, seed: Optional[int] = None) -> DialOpOptimizationTask:
        """Sample a DialOp optimization task."""
        random.seed(seed)

        for _ in range(max(MAX_TABLES, 1)):
            table = Table(self.num_assignments, self.num_assignments)
            user_view, user_known_mask = table.get_random_view(self.p_cell_observed)
            assistant_view, assistant_known_mask = table.get_random_view(
                self.p_cell_observed
            )
            user_scale, assistant_scale = random.uniform(1, 10), random.uniform(1, 10)
            # If neither player knows a value, set it to the mean:
            table.values[
                np.logical_not(user_known_mask) & np.logical_not(assistant_known_mask)
            ] = (table.max_val / 2)

            best_assignment, best_assignment_reward = (
                table.find_max_value_known_assignment(
                    [user_known_mask, assistant_known_mask]
                )
            )

            # Filter to find hard problems:
            _, user_max_reward = table.find_max_value_known_assignment(
                [user_known_mask, user_known_mask]
            )
            _, assistant_max_reward = table.find_max_value_known_assignment(
                [assistant_known_mask, assistant_known_mask]
            )
            if (user_max_reward * 1.25 < best_assignment_reward) and (
                assistant_max_reward * 1.25 < best_assignment_reward
            ):
                break

        user_table = format_table(user_view, user_known_mask, user_scale)
        assistant_table = format_table(
            assistant_view, assistant_known_mask, assistant_scale
        )

        return DialOpOptimizationTask(
            table_values=table.values.tolist(),
            solution=best_assignment,
            max_reward=best_assignment_reward,
            user_table=user_table,
            assistant_table=assistant_table,
            user_known_mask=user_known_mask,
            assistant_known_mask=assistant_known_mask,
            user_scale_factor=user_scale,
            assistant_scale_factor=assistant_scale,
        )


def format_table(table: Table, known: np.ndarray, scale: float) -> str:
    """Format the table as a string."""
    table_values: List[List[Union[int, str]]] = table.values.tolist()
    # Replace empty vals with empty string for display
    for i in range(table.num_rows):
        for j in range(table.num_cols):
            if not known[i][j]:
                table_values[i][j] = ""
            else:
                table_values[i][j] = int(table_values[i][j] * scale)
    # Add row headers
    for i in range(len(table_values)):
        table_values[i] = [WORKERS[i], *table_values[i]]
    # Add col headers
    table_values = [["", *TASKS[: table.num_cols]]] + table_values

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows(table_values)
    return output.getvalue().strip()


def create_dataset(num_examples, num_assignments, p_cell_observed, split_name):
    """Create a dataset of DialOp Optimization problems."""
    examples = []

    system_prompt = {
        "role": "system",
        "content": SYSTEM_PROMPT,
    }

    task_generator = DialOpOptimizationTaskGenerator(num_assignments, p_cell_observed)

    for i in range(num_examples):
        task = task_generator.sample()

        extra_info = {k: v for k, v in task.items() if k != "solution"}
        extra_info["split"] = split_name

        data = {
            "data_source": "dialop_optimization",
            "prompt": [system_prompt],
            "env_class": "dialop_optimization",
            "reward_spec": {
                "method": "rule",
                "ground_truth": task["solution"],
            },
            "extra_info": extra_info,
        }
        examples.append(data)

        if i % 10 == 0 and i > 0:
            print(f"Generated {i} examples")

    return Dataset.from_list(examples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--num_assignments",
        type=int,
        default=8,
        help="Number of assignments to make",
    )
    parser.add_argument(
        "--p_cell_observed",
        type=float,
        default=0.4,
        help="Probability of a paper-reviewer fit being observed",
    )
    parser.add_argument(
        "--train_size", type=int, default=10000, help="Number of training examples"
    )
    parser.add_argument(
        "--test_size", type=int, default=200, help="Number of test examples"
    )

    args = parser.parse_args()

    # Generate datasets
    train_dataset = create_dataset(
        args.train_size, args.num_assignments, args.p_cell_observed, "train"
    )
    val_dataset = create_dataset(
        args.test_size, args.num_assignments, args.p_cell_observed, "test"
    )

    # Save datasets
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(output_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(output_dir, "validation.parquet"))

    print(
        f"Generated {args.train_size} training examples and {args.test_size} test examples"
    )
    print(f"Using {args.num_assignments} assignments")
    print(
        f"Using {args.p_cell_observed} probability of a paper-reviewer fit being observed"
    )
    print(f"Saved to {output_dir}")
