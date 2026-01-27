import inspect
from pathlib import Path
import random
import secrets
import sys
from typing import ClassVar
from typing import Sequence

from ._competitives import COMPETITIVE_CONSTRAINTS


class ConstraintGroupMixin:
    """Mixin that derives a constraint group name from the module path."""

    _group_value: ClassVar[str | None] = None

    def __init__(self, *, seed: int | None = None) -> None:
        instruction_seed = seed if seed is not None else secrets.randbits(64)
        self._instruction_rng = random.Random(instruction_seed)
        self._instruction_choice: str | None = None

    @property
    def group(self) -> str:
        cls = self.__class__
        cached = cls._group_value
        if cached is not None:
            return cached

        definition_path = self._resolve_definition_path(cls)
        group = self._extract_group_name(definition_path)
        if group is None:
            group = self._extract_group_from_module(cls.__module__)

        assert group is not None, f"Could not determine constraint group for class {cls.__name__}"
        cls._group_value = group
        return group

    @property
    def competitives(self) -> list[str]:
        constraints = COMPETITIVE_CONSTRAINTS.get(self.__class__.__name__, ())
        return list(constraints)

    @staticmethod
    def _resolve_definition_path(cls: type["ConstraintGroupMixin"]) -> Path | None:
        try:
            return Path(inspect.getfile(cls)).resolve()
        except TypeError:
            module = sys.modules.get(cls.__module__)
            module_path = getattr(module, "__file__", None) if module else None
            return Path(module_path).resolve() if module_path else None

    @staticmethod
    def _extract_group_name(path: Path | None) -> str | None:
        if path is None:
            return None

        parts = path.parts
        for index, part in enumerate(parts):
            if part == "constraints" and index + 1 < len(parts):
                return ConstraintGroupMixin._camelize(parts[index + 1])
        return None

    @staticmethod
    def _camelize(value: str) -> str:
        return "".join(segment.capitalize() for segment in value.split("_") if segment)

    @staticmethod
    def _extract_group_from_module(module_name: str) -> str | None:
        parts = module_name.split(".")
        try:
            idx = parts.index("constraints")
            raw_segment = parts[idx + 1]
        except (ValueError, IndexError):
            if len(parts) >= 2:
                raw_segment = parts[-2]
            elif parts:
                raw_segment = parts[-1]
            else:
                return None
        return ConstraintGroupMixin._camelize(raw_segment)

    def _random_instruction(self, options: Sequence[str]) -> str:
        if not options:
            raise ValueError("At least one instruction option is required.")
        if self._instruction_choice is None:
            self._instruction_choice = self._instruction_rng.choice(list(options))
        return self._instruction_choice
