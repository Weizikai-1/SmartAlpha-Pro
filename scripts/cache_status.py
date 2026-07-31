"""查看本地缓存状态 — 已缓存多少只股票、覆盖多长时间。

运行: python scripts/cache_status.py
"""

import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from smartalpha.data import DataCache
from smartalpha.data import DataLoader


def main():
    cache = DataCache()
    stats = cache.stats()

    print("=" * 55)
    print("  SmartAlpha Pro 本地缓存状态")
    print("=" * 55)
    print(f"  缓存目录: {cache._dir}")
    print(f"  文件数量: {stats['file_count']}")
    print(f"  总大小:   {stats['total_size_mb']:.1f} MB")
    print()

    # 检测可用数据源
    loader = DataLoader()
    print(f"  活跃数据源: {loader.active_source}")
    if hasattr(loader, '_tushare_available') and loader._tushare_available:
        print(f"  Tushare:    ✅ 可用")
    else:
        print(f"  Tushare:    ❌ 不可用 (需配置 Token)")
    if hasattr(loader, '_akshare_available') and loader._akshare_available:
        print(f"  AKShare:    ✅ 可用 (免费)")
    else:
        print(f"  AKShare:    ❌ 不可用 (pip install akshare)")
    print("=" * 55)


if __name__ == "__main__":
    main()
