#!/usr/bin/env python
"""SmartAlpha Pro 真实数据批量下载脚本。

一键下载 A 股全量日线、指数数据、行业分类，存入本地 Parquet 缓存。

用法:
    # 下载全部 A 股 2020-2026 数据
    python scripts/download_data.py

    # 指定日期范围
    python scripts/download_data.py --start 20240101 --end 20260730

    # 仅下载沪深300成分股（快速测试）
    python scripts/download_data.py --index-hs300

    # 仅下载指定股票
    python scripts/download_data.py --codes 000001.SZ,600000.SH,000300.SH

    # 使用 AKShare（免费，无需 Token）
    python scripts/download_data.py --source akshare

    # 仅更新增量（最近 N 天）
    python scripts/download_data.py --incremental 30

数据源:
    - Tushare (默认): 需要 TUSHARE_TOKEN，快速但有积分限制
    - AKShare: 免费无限制，但较慢（约 3-5 秒/只股票）

输出目录结构:
    data/
    ├── cache/              # 原始日线数据 (Parquet)
    │   ├── daily_000001.SZ.parquet
    │   ├── daily_000002.SZ.parquet
    │   └── ...
    ├── panel_all.parquet   # 合并后的面板数据
    ├── index_market.parquet # 指数日线
    └── industry_map.parquet # 行业分类映射
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smartalpha.data.fetcher import TushareFetcher, TushareError
from smartalpha.data.akshare import AKShareFetcher
from smartalpha.data.index_fetcher import IndexFetcher, BENCHMARK_INDICES
from smartalpha.data.industry_fetcher import (
    IndustryFetcher, save_industry_map_to_cache,
)
from smartalpha.data.panel_builder import PanelBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download")


# ============================================================================
# 配置
# ============================================================================

DEFAULT_DATA_DIR = "data"
DEFAULT_CACHE_DIR = "data/cache"
DEFAULT_START = "20200101"
DEFAULT_END = datetime.now().strftime("%Y%m%d")

# HS300 成分股（简化版，约300只）
HS300_SAMPLE = [
    "600519.SH", "000858.SZ", "601318.SH", "600036.SH", "000333.SZ",
    "601166.SH", "600900.SH", "002415.SZ", "300750.SZ", "000001.SZ",
    "600030.SH", "000002.SZ", "601398.SH", "601288.SH", "600276.SH",
    "002475.SZ", "600887.SH", "603259.SH", "000651.SZ", "002594.SZ",
    "600309.SH", "601012.SH", "000725.SZ", "002714.SZ", "601888.SH",
    "600809.SH", "000568.SZ", "002142.SZ", "601088.SH", "600048.SH",
    "600585.SH", "000063.SZ", "300059.SZ", "002271.SZ", "600028.SH",
    "601857.SH", "600690.SH", "000338.SZ", "002352.SZ", "600009.SH",
]


# ============================================================================
# 核心下载逻辑
# ============================================================================

def download_stock_list(
    source: str = "tushare", data_dir: str = DEFAULT_DATA_DIR
) -> pd.DataFrame:
    """下载 A 股股票列表（含退市+暂停，消除幸存者偏差）。

    Returns:
        股票列表 DataFrame (ts_code, name, industry, list_date, list_status)。
    """
    logger.info("=" * 50)
    logger.info("Step 1/4: 下载股票列表")

    if source == "akshare":
        fetcher = AKShareFetcher()
        df = fetcher.stock_list()
    else:
        try:
            fetcher = TushareFetcher()
            df = fetcher.stock_list()
        except TushareError as e:
            logger.warning(f"Tushare 不可用: {e}，切换到 AKShare")
            fetcher = AKShareFetcher()
            df = fetcher.stock_list()

    if df.empty:
        raise RuntimeError("无法获取股票列表")

    logger.info(f"  获得 {len(df)} 只股票 (含退市/暂停)")
    return df


def download_daily_batch(
    ts_codes: list[str],
    start_date: str,
    end_date: str,
    source: str = "tushare",
    cache_dir: str = DEFAULT_CACHE_DIR,
    incremental: int = 0,
) -> int:
    """批量下载日线数据。

    Args:
        ts_codes: 股票代码列表。
        start_date: 开始日期。
        end_date: 结束日期。
        source: 数据源。
        cache_dir: 缓存目录。
        incremental: 增量更新天数 (0=全量)。

    Returns:
        成功下载的股票数。
    """
    logger.info("=" * 50)
    logger.info(f"Step 2/4: 下载日线数据 ({len(ts_codes)} 只)")

    os.makedirs(cache_dir, exist_ok=True)
    success_count = 0
    skip_count = 0
    fail_count = 0

    if source == "akshare":
        ak_fetcher = AKShareFetcher()
    else:
        try:
            ts_fetcher = TushareFetcher()
        except TushareError as e:
            logger.warning(f"Tushare 不可用: {e}，切换到 AKShare")
            source = "akshare"
            ak_fetcher = AKShareFetcher()

    start_time = time.time()

    for i, code in enumerate(ts_codes):
        cache_path = os.path.join(cache_dir, f"daily_{code}.parquet")

        # 增量模式: 检查缓存，只补充最新 N 天
        if incremental > 0 and os.path.exists(cache_path):
            try:
                existing = pd.read_parquet(cache_path)
                if "trade_date" in existing.columns:
                    last_date = str(existing["trade_date"].max())
                    if last_date >= end_date:
                        skip_count += 1
                        continue
                    # 增量从最后日期+1开始
                    inc_start = (
                        pd.to_datetime(last_date) + timedelta(days=1)
                    ).strftime("%Y%m%d")
                    start_date = inc_start
            except Exception:
                logger.debug(f"  缓存日期读取失败: {cache_path}")

        try:
            if source == "akshare":
                df = ak_fetcher.daily(code, start_date, end_date)
            else:
                df = ts_fetcher.daily(code, start_date, end_date)

            if not df.empty:
                # 合并增量
                if incremental > 0 and os.path.exists(cache_path):
                    try:
                        old = pd.read_parquet(cache_path)
                        df = pd.concat([old, df], ignore_index=True)
                        df = df.drop_duplicates(subset=["trade_date"])
                    except Exception:
                        logger.warning(f"  缓存合并失败，使用新数据: {cache_path}")

                df.to_parquet(cache_path, index=False)
                success_count += 1

        except Exception as e:
            fail_count += 1
            if fail_count <= 3:
                logger.warning(f"  {code} 失败: {e}")

        # 进度输出 (每 50 只)
        if (i + 1) % 50 == 0 or i == len(ts_codes) - 1:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(ts_codes) - i - 1) / rate if rate > 0 else 0
            logger.info(
                f"  进度: {i + 1}/{len(ts_codes)} "
                f"({success_count}成功/{skip_count}跳过/{fail_count}失败) "
                f"| {rate:.1f}只/分钟 "
                f"| ETA: {eta/60:.0f}分钟"
            )

        # API 限速保护
        if source != "akshare" and i < len(ts_codes) - 1:
            time.sleep(0.2)

    logger.info(f"  完成: {success_count} 只成功, {skip_count} 只跳过, {fail_count} 只失败")
    return success_count


def download_index_data(
    start_date: str,
    end_date: str,
    source: str = "tushare",
    data_dir: str = DEFAULT_DATA_DIR,
) -> int:
    """下载基准指数数据。

    Returns:
        成功获取的指数数量。
    """
    logger.info("=" * 50)
    logger.info("Step 3/4: 下载指数数据")

    fetcher = IndexFetcher()
    results = fetcher.fetch_all_benchmarks(start_date, end_date)

    if not results:
        logger.warning("  指数数据获取失败（Tushare 积分不足或网络问题）")
        return 0

    # 保存到单个文件
    all_idx = []
    for code, df in results.items():
        if not df.empty:
            all_idx.append(df)
            logger.info(f"  {code} ({BENCHMARK_INDICES.get(code, '')}): {len(df)} 条")

    if all_idx:
        combined = pd.concat(all_idx, ignore_index=True)
        output = os.path.join(data_dir, "index_market.parquet")
        combined.to_parquet(output, index=False)
        logger.info(f"  已保存至 {output}")
        return len(results)

    return 0


def download_industry_data(
    source: str = "tushare",
    data_dir: str = DEFAULT_DATA_DIR,
) -> dict:
    """下载行业分类数据。

    Returns:
        {stock_code: industry_name} 字典。
    """
    logger.info("=" * 50)
    logger.info("Step 4/4: 下载行业分类")

    fetcher = IndustryFetcher()
    industry_map = fetcher.build_map(prefer_tushare=(source != "akshare"))

    if industry_map:
        output = os.path.join(data_dir, "industry_map.parquet")
        save_industry_map_to_cache(industry_map, output)
        # 统计行业分布
        industries = set(industry_map.values())
        logger.info(f"  行业数: {len(industries)}, 股票数: {len(industry_map)}")
        logger.info(f"  已保存至 {output}")
    else:
        logger.warning("  行业分类获取失败")

    return industry_map


# ============================================================================
# 面板构建
# ============================================================================

def build_final_panel(
    data_dir: str = DEFAULT_DATA_DIR,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Optional[pd.DataFrame]:
    """将缓存中的数据合并为最终面板。"""
    logger.info("=" * 50)
    logger.info("构建最终面板...")

    cache_pattern = os.path.join(cache_dir, "daily_*.parquet")
    import glob
    files = sorted(glob.glob(cache_pattern))

    if not files:
        logger.warning("  无缓存数据，跳过面板构建")
        return None

    frames = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            if not df.empty:
                frames.append(df)
        except Exception:
            logger.warning(f"  读取缓存文件失败: {f.name}")

    if not frames:
        return None

    raw = pd.concat(frames, ignore_index=True)
    builder = PanelBuilder(raw)
    panel = builder.build()

    # 保存面板
    output = os.path.join(data_dir, "panel_all.parquet")
    panel.reset_index().to_parquet(output, index=False)
    logger.info(f"  面板已保存至 {output}")

    # 打印诊断
    print()
    print(builder.diagnose())
    return panel


# ============================================================================
# 主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SmartAlpha Pro 真实数据批量下载",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/download_data.py                          # 全量下载
  python scripts/download_data.py --index-hs300            # 仅沪深300
  python scripts/download_data.py --source akshare         # 使用免费AKShare
  python scripts/download_data.py --incremental 30         # 增量30天
  python scripts/download_data.py --codes 000001.SZ,600000.SH
        """,
    )

    parser.add_argument(
        "--start", default=DEFAULT_START,
        help=f"开始日期 YYYYMMDD (默认: {DEFAULT_START})",
    )
    parser.add_argument(
        "--end", default=DEFAULT_END,
        help=f"结束日期 YYYYMMDD (默认: 今天)",
    )
    parser.add_argument(
        "--source", choices=["tushare", "akshare"], default="tushare",
        help="数据源 (默认: tushare)",
    )
    parser.add_argument(
        "--data-dir", default=DEFAULT_DATA_DIR,
        help=f"数据输出目录 (默认: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--incremental", type=int, default=0,
        help="增量更新天数 (0=全量)",
    )
    parser.add_argument(
        "--index-hs300", action="store_true",
        help="仅下载沪深300成分股 (快速测试)",
    )
    parser.add_argument(
        "--codes", type=str, default="",
        help="指定股票代码，逗号分隔 (如 000001.SZ,600000.SH)",
    )
    parser.add_argument(
        "--skip-index", action="store_true",
        help="跳过多指数下载",
    )
    parser.add_argument(
        "--skip-industry", action="store_true",
        help="跳过行业分类下载",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  SmartAlpha Pro — 真实数据下载工具")
    print("=" * 60)
    print(f"  数据源: {args.source}")
    print(f"  日期范围: {args.start} → {args.end}")
    print(f"  输出目录: {args.data_dir}")
    print(f"  增量模式: {'是 (' + str(args.incremental) + '天)' if args.incremental > 0 else '否 (全量)'}")
    print()

    os.makedirs(args.data_dir, exist_ok=True)
    cache_dir = os.path.join(args.data_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    total_start = time.time()

    # Step 1: 股票列表
    stock_df = download_stock_list(args.source, args.data_dir)
    if stock_df.empty:
        logger.error("无法获取股票列表，退出")
        return 1

    # 筛选股票
    if args.codes:
        codes = [c.strip() for c in args.codes.split(",")]
        logger.info(f"  仅下载指定股票: {codes}")
    elif args.index_hs300:
        codes = HS300_SAMPLE
        logger.info(f"  仅下载沪深300成分股 ({len(codes)} 只)")
    else:
        # 全量: 上市+暂停状态
        codes = stock_df[stock_df["list_status"].isin(["L", "P"])]["ts_code"].tolist()
        # 限制数量 (AKShare 太慢)
        if args.source == "akshare" and len(codes) > 100:
            logger.warning(f"  AKShare 全量下载太慢({len(codes)}只)，限制为前100只")
            logger.warning(f"  建议: 使用 --index-hs300 或 --codes 指定股票")
            codes = codes[:100]
        logger.info(f"  全量下载: {len(codes)} 只股票")

    # Step 2: 日线数据
    n_success = download_daily_batch(
        codes, args.start, args.end,
        source=args.source,
        cache_dir=cache_dir,
        incremental=args.incremental,
    )

    # Step 3: 指数数据
    if not args.skip_index:
        n_index = download_index_data(args.start, args.end, args.source, args.data_dir)
    else:
        n_index = 0

    # Step 4: 行业分类
    if not args.skip_industry:
        industry_map = download_industry_data(args.source, args.data_dir)
    else:
        industry_map = {}

    # 构建面板
    if n_success > 0:
        panel = build_final_panel(args.data_dir, cache_dir)
    else:
        logger.error("  没有成功下载任何股票数据")
        return 1

    # 总结
    total_elapsed = time.time() - total_start
    print()
    print("=" * 60)
    print("  下载完成!")
    print(f"  总耗时: {total_elapsed/60:.1f} 分钟")
    print(f"  股票数据: {n_success} 只")
    print(f"  指数数据: {n_index} 个")
    print(f"  行业分类: {len(industry_map)} 只")
    print(f"  数据目录: {os.path.abspath(args.data_dir)}")
    print()
    print("  下一步:")
    print(f"    python tests/test_real_data.py  # 验证数据质量")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
