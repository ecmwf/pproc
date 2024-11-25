from abc import ABC, abstractmethod
from typing import Annotated, Any, List, Literal, Optional, Tuple, Union

import numexpr
import numpy as np
from pydantic import BaseModel, Field, model_validator

from earthkit.meteo.wind import direction


class Preprocessing(BaseModel, ABC):
    output: Optional[dict] = None

    @abstractmethod
    def apply(
        self, metadata: List[dict], data: List[np.ndarray]
    ) -> Tuple[List[dict], List[np.ndarray]]:
        pass

    def _update_metadata(self, metadata: dict) -> dict:
        new_metadata = metadata.copy()
        if self.output is not None:
            new_metadata.update(self.output)
        return new_metadata


class Scaling(Preprocessing):
    #   - operation: scale
    #     value: 3600
    operation: Literal["scale"] = "scale"
    value: float

    def apply(
        self, metadata: List[dict], data: List[np.ndarray]
    ) -> Tuple[List[dict], List[np.ndarray]]:
        return (
            [self._update_metadata(md) for md in metadata],
            [arr * self.value for arr in data],
        )


class Combination(Preprocessing):
    # - operation: norm
    operation: Literal["direction", "norm", "sum"]

    def apply(
        self, metadata: List[dict], data: List[np.ndarray]
    ) -> Tuple[List[dict], List[np.ndarray]]:
        if self.operation == "norm":
            res = np.linalg.norm(data, axis=0)
        elif self.operation == "direction":
            assert len(data) == 2, "'direction' requires exactly 2 input fields"
            res = direction(data[0], data[1], convention="meteo", to_positive=True)
        elif self.operation == "sum":
            res = np.sum(data, axis=0)
        else:
            assert self.operation in ["direction", "norm"]

        new_metadata = self._update_metadata(metadata[0])
        return [new_metadata], [res]


def find_matching(select: dict, candidates: List[dict]) -> int:
    no_match = object
    for i, c in enumerate(candidates):
        found = True
        for key, val in select.items():
            if c.get(key, no_match) != val:
                found = False
        if found:
            return i
    raise KeyError(f"No data matching {select!r}")


class MaskExpression(BaseModel):
    lhs: Union[float, dict]
    cmp: Literal["<", ">", ">=", "<=", "==", "!="]
    rhs: Union[float, dict]

    @model_validator(mode="before")
    @classmethod
    def validate_model(cls, data: Any) -> Any:
        if isinstance(data, list):
            assert len(data) == 3, "Mask expression should be a [lhs, cmp, rhs] list"
            return {"lhs": data[0], "cmp": data[1], "rhs": data[2]}
        return data

    def _extract(
        self, term: Union[float, dict], metadata: List[dict], data: List[np.ndarray]
    ) -> Union[np.ndarray, float]:
        if isinstance(term, dict):
            idx = find_matching(term, metadata)
            return data[idx]
        return term

    def apply(self, metadata: List[dict], data: List[np.ndarray]) -> np.ndarray:
        expr_data = {
            "lhs": self._extract(self.lhs, metadata, data),
            "rhs": self._extract(self.rhs, metadata, data),
        }
        return numexpr.evaluate("lhs " + self.cmp + " rhs", local_dict=expr_data)


class Masking(Preprocessing):
    #   - operation: mask
    #     select: {param: 228036}
    #     mask: [{param: 228035, level: 250}, ">=", 10]
    operation: Literal["mask"] = "mask"
    mask: MaskExpression
    select: dict
    replacement: float = 0.0

    def apply(
        self, metadata: List[dict], data: List[np.ndarray]
    ) -> Tuple[List[dict], List[np.ndarray]]:
        comp = self.mask.apply(metadata, data)
        idx = find_matching(self.select, metadata)
        masked = np.where(comp, self.replacement, data[idx])
        new_metadata = self._update_metadata(metadata[idx])
        return [new_metadata], [masked]


class PreprocessingConfig(BaseModel):
    # preprocessing:
    #   - operation: norm
    #   - operation: mask
    #     select: {param: 228036}
    #     mask: [{param: 228035, level: 250}, ">=", 10]
    #   - operation: scale
    #     value: 3600
    actions: List[
        Annotated[
            Union[Scaling, Combination, Masking], Field(discriminator="operation")
        ]
    ]

    @model_validator(mode="before")
    @classmethod
    def validate_model(cls, data: Any) -> Any:
        if isinstance(data, list):
            return {"actions": data}
        return data

    def apply(
        self, metadata: List[dict], data: List[np.ndarray]
    ) -> Tuple[List[dict], List[np.ndarray]]:
        for proc in self.actions:
            metadata, data = proc.apply(metadata, data)
        return metadata, data
