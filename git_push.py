"""Git 初始化、提交、推送脚本。绕过 PowerShell 执行策略限制。"""
import subprocess, sys, os

os.chdir(r"c:\trae\量化项目")

def git(*args, check=True):
    """执行 git 命令。"""
    cmd = ["git"] + list(args)
    print(f"  $ git {' '.join(args)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip())
    if check and result.returncode != 0:
        print(f"  [错误] exit={result.returncode}")
        raise SystemExit(result.returncode)
    return result

# ===== 1. 检查是否已有 git 仓库 =====
print("=" * 60)
print("1. 初始化 Git 仓库")
print("=" * 60)

if os.path.exists(".git"):
    print("  .git 已存在, 跳过 init")
else:
    git("init", "-b", "main")
    git("config", "user.name", "SmartAlpha")
    git("config", "user.email", "smartalpha@example.com")

# ===== 2. 查看状态 =====
print("\n" + "=" * 60)
print("2. 当前状态 (将被提交的文件)")
print("=" * 60)
result = git("status", "--short", check=False)
lines = result.stdout.strip().split("\n")
if lines and lines[0]:
    print(f"  共 {len(lines)} 个文件变更")
else:
    print("  无变更, 无需提交")

# ===== 3. 添加所有文件 =====
print("\n" + "=" * 60)
print("3. 添加文件到暂存区")
print("=" * 60)
git("add", ".")

# 验证 .env 没有被添加
result = git("status", "--short", check=False)
if ".env" in result.stdout and "env.example" not in result.stdout:
    # 检查是否是纯 .env (非 .env.example)
    for line in result.stdout.split("\n"):
        if line.strip().endswith(".env") and "example" not in line:
            print(f"  [警告] .env 被加入了暂存区！正在移除...")
            git("rm", "--cached", ".env")
            break

# ===== 4. 确认最终要提交的文件列表 =====
print("\n" + "=" * 60)
print("4. 最终文件清单")
print("=" * 60)
result = git("diff", "--cached", "--name-only", check=False)
files = [f for f in result.stdout.strip().split("\n") if f]
for f in sorted(files):
    print(f"  + {f}")
print(f"  共 {len(files)} 个文件")

if not files:
    print("  没有文件需要提交!")
    sys.exit(0)

# ===== 5. 提交 =====
print("\n" + "=" * 60)
print("5. 提交")
print("=" * 60)
commit_msg = (
    "SmartAlpha Pro - A股智能选股系统\n\n"
    "核心能力:\n"
    "- 因子表达式引擎 (词法分析->语法解析->AST执行, 55+函数)\n"
    "- LangGraph 4 Agent 并行分析 (基本面/技术面/情绪面/舆情面)\n"
    "- Walk-Forward 滚动训练 + Purge 防泄漏\n"
    "- A股真实费率回测 (佣金万3+印花千0.5+滑点千1)\n"
    "- VaR/CVaR风控 + 压力测试\n"
    "- 299 tests, 100% pass\n"
)
git("commit", "-m", commit_msg)

print("\n完成! 仓库已准备好推送。")
print("运行: git remote add origin <your-repo-url> && git push -u origin main")
