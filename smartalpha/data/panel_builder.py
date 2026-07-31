"""面板构建器 — 原始 OHLCV 数据 → MultiIndex 面板。

作为数据层与因子/模型/回测层之间的桥梁：
- 输入: 多股票日线 DataFrame (trade_date, ts_code, OHLCV)
- 输出: MultiIndex DataFrame (date × stock) + 辅助数据

核心功能:
1. 构建 price 面板 (close unstack)
2. 计算后复权价格 (adj_factor)
3. 构建前向收益率
4. 对齐因子数据
5. 面板数据概览与诊断
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from smartalpha._constants import EPS, TRADING_DAYS_PER_YEAR

logger = logging.getLogger(__name__)


class PanelBuilder:
    """面板构建器 — 将原始日线数据转换为 MultiIndex 面板。

    使用示例:
        # 加载原始数据
        raw = pd.read_parquet("data/daily_all.parquet")
        builder = PanelBuilder(raw)
        panel = builder.build()
        # panel 是 MultiIndex(date, ts_code) 的 DataFrame
        # panel["close"].unstack("ts_code") 得到 date×stock 收盘价矩阵
    """

    def __init__(
        self,
        raw_data: pd.DataFrame,
        price_col: str = "close",
        adj_method: str = "backward",
        copy_data: bool = False,
    ) -> None:
        """初始化。

        Args:
            raw_data: 原始日线数据，需含 trade_date, ts_code, OHLCV 列。
            price_col: 价格列名（默认 "close"）。
            adj_method: 复权方式 "backward"(后复权) 或 "forward"(前复权)。
                       强烈推荐后复权，避免前复权的 look-ahead bias。
            copy_data: 是否深拷贝原始数据。默认 False 可节省大量内存。
        """
        self.raw = raw_data.copy() if copy_data else raw_data
        self.price_col = price_col
        self.adj_method = adj_method
        self._panel_cache: pd.DataFrame | None = None

        # 确保日期格式
        if "trade_date" in self.raw.columns:
            self.raw["trade_date"] = self.raw["trade_date"].astype(str)

    # ------------------------------------------------------------------
    # 面板构建
    # ------------------------------------------------------------------

    def build(self) -> pd.DataFrame:
        """构建 MultiIndex 面板（结果缓存，多次调用复用）。

        Returns:
            MultiIndex DataFrame with index (trade_date, ts_code)，
            列: open, high, low, close, vol, amount, adj_factor(可选)
        """
        if self._panel_cache is not None:
            return self._panel_cache

        df = self.raw

        # 标准化日期
        df = df.copy()  # 仅复制一次，避免污染原始数据
        df["date_dt"] = pd.to_datetime(df["trade_date"])

        if "adj_factor" in df.columns:
            # 后复权: 原始价格 × adj_factor
            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    df[f"{col}_adj"] = df[col] * df["adj_factor"]

        # 构建 MultiIndex
        panel = df.set_index(["date_dt", "ts_code"]).sort_index()

        # 添加诊断信息
        panel.attrs["n_stocks"] = panel.index.get_level_values("ts_code").nunique()
        panel.attrs["n_dates"] = panel.index.get_level_values("date_dt").nunique()
        panel.attrs["date_range"] = (
            str(panel.index.get_level_values("date_dt").min().date()),
            str(panel.index.get_level_values("date_dt").max().date()),
        )

        self._panel_cache = panel
        return panel

    # ------------------------------------------------------------------
    # 便捷方法
    # ------------------------------------------------------------------

    def price_matrix(self, use_adj: bool = True) -> pd.DataFrame:
        """构建 date×stock 收盘价矩阵。

        Args:
            use_adj: 是否使用后复权价格（需要 adj_factor 列）。

        Returns:
            DataFrame: index=date, columns=stock_code, values=price。
        """
        panel = self.build()
        col = f"{self.price_col}_adj" if (use_adj and f"{self.price_col}_adj" in panel.columns) else self.price_col
        return panel[col].unstack("ts_code")

    def volume_matrix(self) -> pd.DataFrame:
        """构建 date×stock 成交量矩阵。"""
        panel = self.build()
        if "vol" in panel.columns:
            return panel["vol"].unstack("ts_code")
        return pd.DataFrame()

    def forward_returns(
        self,
        periods: list[int] | None = None,
        purge_days: int = 0,
        use_adj: bool = True,
    ) -> pd.DataFrame:
        """构建前向收益率面板。

        Args:
            periods: 持有期列表。
            purge_days: 清空间隔。
            use_adj: 使用后复权价格。

        Returns:
            MultiIndex DataFrame: (trade_date, ts_code) × (ret_1d, ret_5d, ...)
        """
        if periods is None:
            periods = [1, 5, 10]

        pw = self.price_matrix(use_adj=use_adj)
        all_ret = {}

        for p in periods:
            col_name = f"ret_{p}d"
            if purge_days > 0:
                shifted = pw.shift(-purge_days)
                ret = shifted.pct_change(periods=p).shift(-p)
            else:
                ret = pw.pct_change(periods=p).shift(-p)
            all_ret[col_name] = ret.stack()

        result = pd.DataFrame(all_ret)
        result.index.names = ["trade_date", "ts_code"]
        return result

    def stock_list(self) -> np.ndarray:
        """获取面板中的股票列表。"""
        panel = self.build()
        return panel.index.get_level_values("ts_code").unique().values

    def date_list(self) -> pd.DatetimeIndex:
        """获取面板中的交易日列表。"""
        panel = self.build()
        return panel.index.get_level_values("date_dt").unique().sort_values()

    # ------------------------------------------------------------------
    # 诊断
    # ------------------------------------------------------------------

    def diagnose(self) -> str:
        """生成面板数据诊断报告。

        Returns:
            格式化的诊断字符串。
        """
        panel = self.build()
        n_stocks = panel.attrs.get("n_stocks", 0)
        n_dates = panel.attrs.get("n_dates", 0)
        date_range = panel.attrs.get("date_range", ("?", "?"))

        pw = self.price_matrix(use_adj=False)
        missing_ratio = pw.isnull().mean().mean() if not pw.empty else 1.0

        lines = [
            "=" * 50,
            "面板数据诊断报告",
            "=" * 50,
            f"股票数: {n_stocks}",
            f"交易日数: {n_dates}",
            f"日期范围: {date_range[0]} → {date_range[1]}",
            f"缺失率: {missing_ratio:.2%}",
            "",
            "--- 列信息 ---",
        ]
        for col in panel.columns:
            dtype_str = str(panel[col].dtype)
            null_count = panel[col].isnull().sum()
            lines.append(f"  {col}: {dtype_str}, NaN={null_count}")

        # 日均交易股票数
        daily_count = pw.notnull().sum(axis=1)
        lines.append("")
        lines.append("--- 日均有效股票数 ---")
        lines.append(f"  均值: {daily_count.mean():.0f}")
        lines.append(f"  最小: {daily_count.min()}")
        lines.append(f"  最大: {daily_count.max()}")

        return "\n".join(lines)


# ============================================================================
# 便捷函数
# ============================================================================

def build_panel_from_cache(
    cache_dir: str = "data/cache",
    start_date: str = "20200101",
    end_date: str = "20260730",
) -> pd.DataFrame:
    """从本地缓存目录加载并构建面板。

    遍历 data/cache/ 下的所有 Parquet 文件，合并为统一面板。

    Args:
        cache_dir: 缓存目录路径。
        start_date: 开始日期 YYYYMMDD。
        end_date: 结束日期 YYYYMMDD。

    Returns:
        MultiIndex 面板 DataFrame。
    """
    import os
    import glob

    pattern = os.path.join(cache_dir, "daily_*.parquet")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"缓存目录 {cache_dir} 为空，请先运行数据下载脚本")

    frames = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            if "trade_date" in df.columns:
                df["trade_date"] = df["trade_date"].astype(str)
                mask = (df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)
                df = df[mask]
            if not df.empty:
                frames.append(df)
        except Exception as e:
            logger.warning(f"读取 {f} 失败: {e}")

    if not frames:
        raise ValueError("没有有效的缓存数据")

    raw = pd.concat(frames, ignore_index=True)
    builder = PanelBuilder(raw)
    panel = builder.build()
    return panel


def align_factor_to_panel(
    factor_df: pd.DataFrame,
    panel: pd.DataFrame,
    fillna: float = 0.0,
) -> pd.DataFrame:
    """将因子数据对齐到面板的 (date, stock) 索引。

    Args:
        factor_df: 因子数据 (支持宽表 date×stock 或 MultiIndex)。
        panel: 参考面板。
        fillna: 缺失值填充。

    Returns:
        对齐后的因子 DataFrame。
    """
    if isinstance(factor_df.index, pd.MultiIndex):
        factor_aligned = factor_df.reindex(panel.index)
    else:
        # 宽表: 尝试对齐
        pw = panel["close"].unstack("ts_code") if "close" in panel.columns else panel.unstack("ts_code")
        common_stocks = [c for c in factor_df.columns if c in pw.columns]
        common_dates = pw.index.intersection(factor_df.index)
        aligned = factor_df.loc[common_dates, common_stocks]
        factor_aligned = aligned.stack()
        factor_aligned.index.names = ["trade_date", "ts_code"]

    if fillna is not None:
        factor_aligned = factor_aligned.fillna(fillna)

    return factor_aligned
