"""章节/目录类查询路由单元测试。"""

from src.utils.query_routing import is_chapter_query


def test_chapter_queries_detected():
    assert is_chapter_query("第一章“集合与常用逻辑用语”包含哪些小节？")
    assert is_chapter_query("第三章 函数的概念与性质有哪些小节？")
    assert is_chapter_query("各章末尾有哪些复习栏目？")
    assert is_chapter_query("教材中有哪些“阅读与思考”栏目？")
    assert is_chapter_query("人教A版必修一包含哪五章内容？")


def test_non_chapter_queries_not_detected():
    assert not is_chapter_query("什么是集合？")
    assert not is_chapter_query("函数的三要素指什么？")
    assert not is_chapter_query("方程 x^2 = x 的所有实数根组成的集合是什么？")
    assert not is_chapter_query("基本不等式是什么？")
