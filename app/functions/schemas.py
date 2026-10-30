from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.config import get_settings
from app.functions.parser import ExpressionError, check_expression

Interval = Annotated[list[float], Field(min_length=2, max_length=2)]
Domain = Annotated[list[Interval], Field(min_length=3, max_length=3)]


def validate_domain(domain: list[list[float]]) -> list[list[float]]:
    for axis, (lower, upper) in zip("xyz", domain, strict=True):
        if lower >= upper:
            raise ValueError(f"El intervalo del eje {axis} debe cumplir min < max")
    return domain


def validate_resolution(resolution: int) -> int:
    settings = get_settings()
    if not settings.min_resolution <= resolution <= settings.max_resolution:
        raise ValueError(
            f"La resolución debe estar entre {settings.min_resolution} y {settings.max_resolution}"
        )
    return resolution


def validate_expression(expression: str) -> str:
    expression = expression.strip()
    if not expression:
        raise ValueError("La expresión no puede estar vacía")
    if len(expression) > get_settings().max_expression_length:
        raise ValueError("La expresión es demasiado larga")
    try:
        check_expression(expression)
    except ExpressionError as error:
        raise ValueError(str(error)) from error
    return expression


class FunctionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    expression: str
    domain: Domain
    resolution: int = Field(default_factory=lambda: get_settings().default_resolution)

    _check_expression = field_validator("expression")(validate_expression)
    _check_domain = field_validator("domain")(validate_domain)
    _check_resolution = field_validator("resolution")(validate_resolution)


class FunctionUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    expression: str | None = None
    domain: Domain | None = None
    resolution: int | None = None

    @field_validator("expression")
    @classmethod
    def check_expression(cls, value: str | None) -> str | None:
        return None if value is None else validate_expression(value)

    @field_validator("domain")
    @classmethod
    def check_domain(cls, value: list[list[float]] | None) -> list[list[float]] | None:
        return None if value is None else validate_domain(value)

    @field_validator("resolution")
    @classmethod
    def check_resolution(cls, value: int | None) -> int | None:
        return None if value is None else validate_resolution(value)


class FunctionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    owner_id: int
    name: str
    expression: str
    domain: list[list[float]]
    resolution: int
    version: int
    created_at: datetime
    updated_at: datetime


class PointEvaluation(BaseModel):
    x: float
    y: float
    z: float
    value: float
    is_inside: bool
class PointEvaluation(BaseModel):
    x: float
    y: float
    z: float
    value: float | None = Field(description="Valor de f; null si no es un número finito")
    is_finite: bool
    is_inside: bool | None = Field(description="f <= 0; null si f no está definida (NaN)")