import re

import numpy as np
import pytest

from app.functions.parser import ExpressionError, check_expression, compile_expression


def test_compiled_expression_evaluates_arrays():
    evaluate = compile_expression("x**2 + y**2 + z**2 - 1")
    values = evaluate(np.array([0.0, 1.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0]))

    np.testing.assert_allclose(values, [-1.0, 0.0])


def test_constant_expression_is_broadcast():
    evaluate = compile_expression("pi")
    values = evaluate(np.zeros(4), np.zeros(4), np.zeros(4))

    assert values.shape == (4,)


@pytest.mark.parametrize("expression", ["sin(x) * cos(y) + z", "sqrt(x**2 + y**2) - 0.5", "exp(-x) - e"])
def test_check_accepts_allowed_functions(expression):
    check_expression(expression)


@pytest.mark.parametrize("expression", ["x +* y", "open('/etc/passwd')", "__import__('os')", "foo(x)"])
def test_check_rejects_invalid_expressions(expression):
    with pytest.raises(ExpressionError):
        check_expression(expression)



@pytest.mark.parametrize(
    "expression",
    [
        "x**2 + y**2 + z**2 - 1",
        "(sqrt(x**2 + y**2) - 1)**2 + z**2 - 0.25**2",
        "sin(x)*cos(y) + sin(y)*cos(z) + sin(z)*cos(x)",
        "1 - 1/(x**2 + y**2 + z**2)",
        "tan(x) - z",
        "1/x - y",
        "-x + abs(y) - log(2) * pi",
    ],
)
def test_check_accepts_readme_examples(expression):
    check_expression(expression)


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("().__class__.__base__.__subclasses__().__len__() * 0 + x", "Función no permitida"),
        ("x.real", "Construcción no permitida"),
        ("lambda: 1", "Construcción no permitida"),
        ("x if y else z", "Construcción no permitida"),
        ("x < y", "Construcción no permitida"),
        ("[x][0]", "Construcción no permitida"),
        ("'texto'", "Constante no permitida"),
        ("True", "Constante no permitida"),
        ("1j", "Constante no permitida"),
        ("x // 2", "Operador no permitido: //"),
        ("x % 2", "Operador no permitido: %"),
        ("x^2", "para potencias usa **"),
        ("not x", "Operador no permitido: not"),
        ("sin(x, y)", "exactamente un argumento"),
        ("sin", "Nombre no permitido: sin"),
        ("w + 1", "Nombre no permitido: w"),
    ],
)
def test_check_rejects_forbidden_constructions(expression, message):
    with pytest.raises(ExpressionError, match=re.escape(message)):
        check_expression(expression)