from typing import Awaitable
from typing import Protocol
from typing import TypeAlias


ConstraintEvaluation: TypeAlias = tuple[bool, str | None]


class Constraint(Protocol):
    def evaluate(self, value: str) -> ConstraintEvaluation | Awaitable[ConstraintEvaluation]: ...

    def instructions(self, train_or_test: str = "train") -> str: ...

    @property
    def group(self) -> str: ...

    def rewrite_instructions(self) -> str: ...

    @property
    def competitives(self) -> list[str]: ...


class Prompt(Protocol):
    def text(self, constraints: list[Constraint], *, train_or_test: str = "train") -> str: ...

    @property
    def document(self) -> str: ...
