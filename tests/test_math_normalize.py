"""公式文本规范化单元测试。"""

from src.utils.math_normalize import normalize_math_text


def test_exp_braces_collapsed():
    assert normalize_math_text("x^{2}=x") == "x^2=x"
    assert normalize_math_text("a^{m-n}") == "a^m-n"


def test_belong_symbols_unified():
    assert normalize_math_text("$ a \\in A $") == "$ a ∈ A $"
    assert normalize_math_text("$ a \\notin A $") == "$ a ∉ A $"


def test_log_underscore_collapsed():
    assert normalize_math_text("\\log_{a}M") == "log_aM"


def test_whitespace_collapsed():
    assert normalize_math_text("a   b\u3000c") == "a b c"
