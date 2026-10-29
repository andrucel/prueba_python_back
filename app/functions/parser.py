import ast
from collections.abc import Callable
from types import CodeType

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ImplicitCallable = Callable[[FloatArray, FloatArray, FloatArray], FloatArray]

ALLOWED_FUNCTIONS: dict[str, object] = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,
    "abs": np.abs,
}
ALLOWED_CONSTANTS: dict[str, float] = {"pi": np.pi, "e": np.e}
ALLOWED_VARIABLES = frozenset({"x", "y", "z"})
ALLOWED_NAMES: dict[str, object] = {**ALLOWED_FUNCTIONS, **ALLOWED_CONSTANTS}

BINARY_OPERATORS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
UNARY_OPERATORS = (ast.UAdd, ast.USub)

# Nombres legibles para los mensajes de error.
OPERATOR_SYMBOLS: dict[type[ast.AST], str] = {
    ast.FloorDiv: "//",
    ast.Mod: "%",
    ast.MatMult: "@",
    ast.BitXor: "^ (para potencias usa **)",
    ast.BitAnd: "&",
    ast.BitOr: "|",
    ast.LShift: "<<",
    ast.RShift: ">>",
    ast.Invert: "~",
    ast.Not: "not",
}

SAFE_GLOBALS: dict[str, object] = {"__builtins__": {}}

_PROBE = np.linspace(0.25, 0.75, 3)


class ExpressionError(ValueError):
    pass


def _symbol(operator: ast.AST) -> str:
    return OPERATOR_SYMBOLS.get(type(operator), type(operator).__name__)


def _check_node(node: ast.AST) -> None:
    """Recorre el árbol y rechaza cualquier construcción fuera de la lista blanca."""
    if isinstance(node, ast.Expression):
        _check_node(node.body)
    elif isinstance(node, ast.BinOp):
        if not isinstance(node.op, BINARY_OPERATORS):
            raise ExpressionError(f"Operador no permitido: {_symbol(node.op)}")
        _check_node(node.left)
        _check_node(node.right)
    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, UNARY_OPERATORS):
            raise ExpressionError(f"Operador no permitido: {_symbol(node.op)}")
        _check_node(node.operand)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            raise ExpressionError(f"Constante no permitida: {node.value!r}")
    elif isinstance(node, ast.Name):
        if node.id not in ALLOWED_VARIABLES and node.id not in ALLOWED_CONSTANTS:
            raise ExpressionError(f"Nombre no permitido: {node.id}")
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCTIONS:
            raise ExpressionError(f"Función no permitida: {ast.unparse(node.func)}")
        if len(node.args) != 1 or node.keywords:
            raise ExpressionError(f"La función {node.func.id} recibe exactamente un argumento")
        _check_node(node.args[0])
    else:
        raise ExpressionError(f"Construcción no permitida: {type(node).__name__}")


def _compile(expression: str) -> CodeType:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ExpressionError(f"Sintaxis inválida: {error.msg}") from error
    _check_node(tree)
    # Las constantes se evalúan como float: una potencia entera como 9**9**9 se calcularía con
    # precisión arbitraria y bloquearía el proceso; en float desborda al instante.
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            node.value = float(node.value)
    return compile(tree, "<expression>", "eval")


def compile_expression(expression: str) -> ImplicitCallable:
    code = _compile(expression)

    def evaluate(x: FloatArray, y: FloatArray, z: FloatArray) -> FloatArray:
        namespace = {**ALLOWED_NAMES, "x": x, "y": y, "z": z}
        result = eval(code, SAFE_GLOBALS, namespace)  # noqa: S307
        return np.broadcast_to(np.asarray(result, dtype=np.float64), np.broadcast(x, y, z).shape)

    return evaluate


def check_expression(expression: str) -> None:
    """Valida la expresión compilándola y evaluándola sobre unos puntos de prueba."""
    evaluate = compile_expression(expression)
    try:
        evaluate(_PROBE, _PROBE, _PROBE)
    except OverflowError as error:
        raise ExpressionError("La expresión produce un número demasiado grande") from error
    except Exception as error:
        raise ExpressionError(f"No se pudo evaluar la expresión: {error}") from error


def evaluate_point(expression: str, x: float, y: float, z: float) -> float:
    evaluate = compile_expression(expression)
    return float(evaluate(x, y, z))