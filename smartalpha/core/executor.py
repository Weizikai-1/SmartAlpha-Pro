"""AST 执行器模块。

使用访问者模式遍历 AST 并计算因子值。
支持向量化运算（数值数组）与标量运算。

求值策略：
- 变量节点：从上下文（变量 → 数值数组映射）中查找
- 数字节点：直接返回标量值
- 二元运算：逐元素或标量计算
- 函数调用：委托给 FinancialFunctionLibrary
- 条件表达式：逐元素三元运算
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np

from .ast import (
    ASTNode,
    ASTVisitor,
    BinaryOpNode,
    ConditionalNode,
    FuncCallNode,
    NumNode,
    StringNode,
    UnaryOpNode,
    VarNode,
)
from .functions import FinancialFunctionLibrary
from ._func_helpers import to_array


Number = Union[float, int]
Value = Union[Number, "np.ndarray", List[float]]


class ExecutionError(Exception):
    """执行错误异常。"""

    def __init__(self, message: str, node: Optional[ASTNode] = None) -> None:
        self.node = node
        super().__init__(message)


class ASTExecutor(ASTVisitor):
    """AST 执行器。

    通过访问者模式遍历 AST，结合变量上下文与函数库计算最终结果。

    使用示例::

        executor = ASTExecutor(
            variables={"$close": np.array([10.0, 11.0, 12.0])},
            functions=FinancialFunctionLibrary(),
        )
        result = executor.execute(ast_root)

    Attributes:
        variables: 变量名 → 数值数组 的映射。
        functions: 可用的金融函数库。
    """

    def __init__(
        self,
        variables: Optional[Dict[str, Value]] = None,
        functions: Optional[FinancialFunctionLibrary] = None,
    ) -> None:
        """初始化执行器。

        Args:
            variables: 变量 → 值的映射（值可为标量或 numpy 数组）。
            functions: 金融函数库实例，为 None 时自动创建默认实例。
        """
        self.variables: Dict[str, Value] = variables or {}
        self.functions: FinancialFunctionLibrary = (
            functions or FinancialFunctionLibrary()
        )

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def execute(self, node: ASTNode, variables: Optional[Dict[str, Value]] = None) -> Value:
        """执行 AST 并返回计算结果。

        Args:
            node: AST 根节点。
            variables: 变量 → 值的映射（可选，覆盖构造函数传入的值）。

        Returns:
            计算结果（标量、数组或列表）。
        """
        if variables is not None:
            self.variables = variables
        result = node.accept(self)
        return self._to_native(result)

    # ------------------------------------------------------------------
    # 访问者方法实现
    # ------------------------------------------------------------------

    def visit_var(self, node: VarNode) -> Value:
        """访问变量节点。"""
        # 先尝试原名称，若无则尝试添加 $ 前缀
        if node.name in self.variables:
            return self.variables[node.name]

        alt_name = node.name if not node.name.startswith("$") else f"${node.name}"
        if alt_name in self.variables:
            return self.variables[alt_name]

        raise ExecutionError(f"未定义的变量: {node.name}", node)

    def visit_num(self, node: NumNode) -> Value:
        """访问数字节点。"""
        return node.value

    def visit_string(self, node: StringNode) -> Value:
        """访问字符串节点。"""
        return node.value

    def visit_binary_op(self, node: BinaryOpNode) -> Value:
        """访问二元运算节点。"""
        left = self.execute(node.left)
        right = self.execute(node.right)

        op = node.op

        # 算术运算
        if op == "+":
            return to_array(left) + to_array(right)
        if op == "-":
            return to_array(left) - to_array(right)
        if op == "*":
            return to_array(left) * to_array(right)
        if op == "/":
            divisor = to_array(right)
            result = to_array(left) / divisor
            result = np.where(np.isinf(result), 0.0, result)
            return np.where(np.isnan(result), 0.0, result)
        if op == "%":
            return to_array(left) % to_array(right)
        if op == "^":
            return np.power(to_array(left), to_array(right))

        # 比较运算
        if op == ">":
            return np.greater(to_array(left), to_array(right)).astype(np.float64)
        if op == ">=":
            return np.greater_equal(to_array(left), to_array(right)).astype(np.float64)
        if op == "<":
            return np.less(to_array(left), to_array(right)).astype(np.float64)
        if op == "<=":
            return np.less_equal(to_array(left), to_array(right)).astype(np.float64)
        if op == "==":
            return np.equal(to_array(left), to_array(right)).astype(np.float64)
        if op == "!=":
            return np.not_equal(to_array(left), to_array(right)).astype(np.float64)

        # 逻辑运算
        if op == "&&":
            al = to_array(left).astype(bool)
            bl = to_array(right).astype(bool)
            return np.logical_and(al, bl).astype(np.float64)
        if op == "||":
            al = to_array(left).astype(bool)
            bl = to_array(right).astype(bool)
            return np.logical_or(al, bl).astype(np.float64)

        raise ExecutionError(f"未知的运算符: {op}", node)

    def visit_unary_op(self, node: UnaryOpNode) -> Value:
        """访问一元运算节点。"""
        operand = self.execute(node.operand)

        if node.op == "-":
            return -to_array(operand)
        if node.op == "!":
            return (~to_array(operand).astype(bool)).astype(float)

        raise ExecutionError(f"未知的一元运算符: {node.op}", node)

    def visit_func_call(self, node: FuncCallNode) -> Value:
        """访问函数调用节点。"""
        func_name = node.name.upper()
        args = [self.execute(arg) for arg in node.args]

        if not self.functions.has_function(func_name):
            raise ExecutionError(f"未知的函数: {node.name}", node)

        return self.functions.call(func_name, *args)

    def visit_conditional(self, node: ConditionalNode) -> Value:
        """访问条件表达式节点（三元 IF）。"""
        cond = to_array(self.execute(node.condition))
        then_val = to_array(self.execute(node.then_branch))
        if node.else_branch is not None:
            else_val = to_array(self.execute(node.else_branch))
        else:
            else_val = np.nan

        cond_bool = cond.astype(bool) if isinstance(cond, np.ndarray) else bool(cond)
        return np.where(cond_bool, then_val, else_val)

    @staticmethod
    def _to_native(result: Value) -> Value:
        """将 numpy 数组转换为更友好的原生类型。"""
        if isinstance(result, np.ndarray):
            if result.ndim == 0:
                return float(result)
            return result
        return result
