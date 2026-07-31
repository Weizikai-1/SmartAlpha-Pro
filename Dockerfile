# ============================================================================
# SmartAlpha Pro v2.0 — 生产级 Docker 镜像
# ============================================================================
# 
# 构建:
#   docker build -t smartalpha-pro .
#
# 运行回测:
#   docker run --rm -v ./data:/app/data --env-file .env smartalpha-pro \
#     python -c "from smartalpha.data.panel_builder import build_panel_from_cache; ..."
#
# 运行测试:
#   docker run --rm -v ./data:/app/data --env-file .env smartalpha-pro \
#     pytest tests/ -v --ignore=tests/test_real_data.py
#
# 交互模式:
#   docker run --rm -it -v ./data:/app/data --env-file .env smartalpha-pro bash
# ============================================================================

FROM python:3.11-slim-bookworm AS builder

# 构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先安装依赖（利用 Docker 层缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ============================================================================
# 生产镜像
# ============================================================================
FROM python:3.11-slim-bookworm

LABEL org.opencontainers.image.title="SmartAlpha Pro"
LABEL org.opencontainers.image.description="Industrial-grade A-share quantitative stock selection system"
LABEL org.opencontainers.image.version="2.0.0"
LABEL org.opencontainers.image.authors="SmartAlpha Team"

# 运行时依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# 从构建阶段复制依赖
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 复制应用代码
COPY --chown=appuser:appuser . .

# 创建数据目录
RUN mkdir -p /app/data/cache && chown -R appuser:appuser /app/data

USER appuser

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import smartalpha; print('OK')" || exit 1

# 默认命令
ENTRYPOINT ["python"]
CMD ["-c", "from smartalpha import __file__; print('SmartAlpha Pro v2.0.0 ready.')"]
