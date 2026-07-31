"""抽象语法树（AST）节点定义模块。

定义因子表达式引擎所需的 AST 节点类型，采用访问者模式支持
后续的执行与分析操作。

节点类型：
- ASTNode:  所有 AST 节点的抽象基类
- VarNode:  变量引用（$VAR_NAME）
- NumNode:  数字字面量
- StringNode: 字符串字面量
- BinaryOpNode: 二元运算（算术 / 比较 / 逻辑）
- UnaryOpNode:  一元运算（-、!）
- FuncCallNode: 函数调用
- ConditionalNode: 条件表达式（IF(cond, then, else)）
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


class ASTNode(ABC):
    """AST 节点抽象基类。

    所有具体节点类型必须实现 `accept` 方法以支持访问者模式。
    """

    @abstractmethod
    def accept(self, visitor: "ASTVisitor") -> object:
        """接受访问者调用（双分派）。

        Args:
            visitor: 访问者实例。

        Returns:
            访问者处理该节点的返回值。
        """
        ...


# ---------------------------------------------------------------------------
# 具体节点类型
# ---------------------------------------------------------------------------

@dataclass
class VarNode(ASTNode):
    """变量引用节点。

    Attributes:
        name: 变量名（含 $ 前缀，如 ``$close``）。
    """

    name: str

    def accept(self, visitor: "ASTVisitor") -> object:
        return visitor.visit_var(self)


@dataclass
class NumNode(ASTNode):
    """数字字面量节点。

    Attributes:
        value: 浮点数值。
    """

    value: float

    def accept(self, visitor: "ASTVisitor") -> object:
        return visitor.visit_num(self)


@dataclass
class StringNode(ASTNode):
    """字符串字面量节点。

    Attributes:
        value: 字符串值（不含引号）。
    """

    value: str

    def accept(self, visitor: "ASTVisitor") -> object:
        return visitor.visit_string(self)


@dataclass
class BinaryOpNode(ASTNode):
    """二元运算节点。

    Attributes:
        op: 运算符字符串（+、-、*、/、%、^、>、<、>=、<=、==、!=、&&、||）。
        left: 左操作数节点。
        right: 右操作数节点。
    """

    op: str
    left: ASTNode
    right: ASTNode

    def accept(self, visitor: "ASTVisitor") -> object:
        return visitor.visit_binary_op(self)


@dataclass
class UnaryOpNode(ASTNode):
    """一元运算节点。

    Attributes:
        op: 运算符字符串（-、!）。
        operand: 操作数节点。
    """

    op: str
    operand: ASTNode

    def accept(self, visitor: "ASTVisitor") -> object:
        return visitor.visit_unary_op(self)


@dataclass
class FuncCallNode(ASTNode):
    """函数调用节点。

    Attributes:
        name: 函数名（如 RANK、ZSCORE）。
        args: 实参节点列表。
    """

    name: str
    args: List[ASTNode] = field(default_factory=list)

    def accept(self, visitor: "ASTVisitor") -> object:
        return visitor.visit_func_call(self)


@dataclass
class ConditionalNode(ASTNode):
    """条件表达式节点（三元 IF）。

    Attributes:
        condition: 条件表达式节点。
        then_branch: 真值分支节点。
        else_branch: 假值分支节点（可选）。
    """

    condition: ASTNode
    then_branch: ASTNode
    else_branch: Optional[ASTNode] = None

    def accept(self, visitor: "ASTVisitor") -> object:
        return visitor.visit_conditional(self)


# ---------------------------------------------------------------------------
# 访问者抽象基类
# ---------------------------------------------------------------------------

class ASTVisitor(ABC):
    """AST 访问者抽象基类。

    定义针对每种节点类型的访问方法，子类按需覆写。
    """

    def visit_var(self, node: VarNode) -> object:
        raise NotImplementedError

    def visit_num(self, node: NumNode) -> object:
        raise NotImplementedError

    def visit_string(self, node: StringNode) -> object:
        raise NotImplementedError

    def visit_binary_op(self, node: BinaryOpNode) -> object:
        raise NotImplementedError

    def visit_unary_op(self, node: UnaryOpNode) -> object:
        raise NotImplementedError

    def visit_func_call(self, node: FuncCallNode) -> object:
        raise NotImplementedError

    def visit_conditional(self, node: ConditionalNode) -> object:
        raise NotImplementedError
