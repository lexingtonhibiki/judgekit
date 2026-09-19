# -*- coding: utf-8 -*-
"""零成本关键词规则供应商：engine.rules_fallback 实际干活，这里只占注册表位。"""


class RulesProvider:
    def __init__(self, name: str = "rules"):
        self.name = name
