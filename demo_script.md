# SmartAlpha Pro + AML 反洗钱 — 5分钟演示脚本

> **录制工具**: OBS Studio (免费) / Windows自带录屏 (Win+G)
> **录制方式**: 提前开好浏览器和终端，按脚本切换窗口
> **时长**: 5分00秒

---

## 第1段：自我介绍 (0:00 - 0:30)

```
画面: VS Code + 两个项目文件夹并排

"我是XX学校金融科技专业的学生。今天展示我的两个毕业项目——
一个面向C端的量化选股系统 SmartAlpha Pro，
一个面向B端的反洗钱检测系统 AML Detection。
两个都是基于 LangGraph 多智能体框架从零搭建的。"
```

## 第2段：SmartAlpha 快速跑起来 (0:30 - 1:30)

```
画面: 终端，cd 到量化项目目录

操作:
cd C:\trae\量化项目
streamlit run smartalpha/ui/app.py

旁白:
"先看量化选股系统。输入000001.SZ平安银行，选标准分析模式，
点击开始分析。
背后是4个LangGraph Agent并行工作——
基本面Agent分析财务因子、
技术面Agent算55种技术指标、
情绪面Agent和舆情面Agent各自推理，
最后辩论合成出买卖建议。"

画面: 浏览器打开 http://localhost:8501
     输入 000001.SZ，点"开始分析"
     等结果出来，鼠标在页面滑动展示：
     1. 决策卡片（看多/看空/中性）
     2. Agent面板（4个Agent的信号和置信度）
     3. 多空辩论结果
     4. 完整Markdown报告
```

## 第3段：展示表达式引擎 (1:30 - 2:10)

```
画面: 切回VS Code，打开 smartalpha/core/functions.py

旁白:
"这个系统的技术核心是我自己写的因子表达式引擎。
用户输入 RANK(DELTA($close,1)) / MEAN($volume,20) 这样的表达式，
系统分四层处理：
词法分析把字符串切成Token，
递归下降解析器生成AST语法树，
最后向量化执行器调用55个金融函数计算结果。
这是从编译原理搬过来的技术栈，不是调库。"

画面: 打开 smartalpha/core/lexer.py → parser.py → ast.py → executor.py
     快速切换四个文件，每个停3秒
```

## 第4段：切到反洗钱系统 (2:10 - 3:10)

```
画面: 终端开新窗口

操作:
cd C:\trae\反洗钱
streamlit run app.py

旁白:
"再看B端反洗钱检测系统。同样是LangGraph架构，但这里用了6个Agent——
数据预处理、10条规则引擎、GNN图分析三个并行跑，
风险分超过70分的交易自动送给DeepSeek做语义深审，
最后生成央行格式的可疑交易报告STR。"

画面: 浏览器打开 http://localhost:8501 (或另一个端口)
     勾选"Demo模式"，点"启动检测"
     等结果出来，滚动展示：
     1. 检测概览（总交易、规则命中、LLM深审数）
     2. Agent流水线（6个Agent各自状态）
     3. 高风险交易详情
     4. STR报告全文
```

## 第5段：GNN + LLM 深度 (3:10 - 3:50)

```
画面: VS Code 打开 gnn_model.py

旁白:
"规则引擎打初筛，GNN做图分析——
把每笔交易抽象成账户节点和有向边，用GAT图注意力网络做节点分类，
判断哪些是欺诈账户。训练时我用pos_weight处理99:1的类别不平衡，
测试集F1到0.9。"

画面: 切到 agents/llm_reviewer.py

旁白:
"高风险交易自动推给DeepSeek LLM做语义分析，
输出结构化JSON——嫌疑等级、洗钱类型、建议措施。
还加了RAG记忆，每次分析完存到案例库，下次遇到类似场景会调历史案例做参考。"
```

## 第6段：API + 收尾 (3:50 - 4:40)

```
画面: VS Code 打开 api.py

旁白:
"系统还提供了FastAPI接口——
/health 健康检查、
/detect POST提交检测任务、
/report 拉取完整报告。
可以直接对接银行内部系统。"

画面: 终端跑一条curl命令

操作:
curl http://localhost:8000/health

旁白:
"两个项目都放在GitHub开源，有完整的README、CI/CD流水线、
Docker配置、还有65个测试用例。
更多细节欢迎看代码或联系我。"

画面: 浏览器打开 GitHub 项目主页 (如果有的话)
     或者 VS Code 展示项目目录结构

旁白:
"谢谢观看。"
```

## 第7段：结束 (4:40 - 5:00)

```
画面: 黑底白字联系方式

"GitHub: [你的GitHub]
 邮箱: [你的邮箱]
 微信: [你的微信]"
```

---

## 录制前的准备工作

### 1. 确保两个项目能跑通

```powershell
# 终端1: 启动 SmartAlpha
cd C:\trae\量化项目
streamlit run smartalpha/ui/app.py

# 终端2: 启动 AML  
cd C:\trae\反洗钱
streamlit run app.py

# 终端3: 启动 AML API (可选，展示用)
cd C:\trae\反洗钱
uvicorn api:app --port 8000
```

### 2. 提前准备好浏览器标签页
- `http://localhost:8501` (SmartAlpha)
- `http://localhost:8502` 或AML的端口
- GitHub项目主页

### 3. VS Code 提前打开的标签
- `smartalpha/core/functions.py`
- `smartalpha/core/lexer.py`
- `gnn_model.py`
- `agents/llm_reviewer.py`
- `api.py`

### 4. 安装 OBS Studio
https://obsproject.com/
- 免费，中文界面
- 设置录制区域为全屏或窗口
- 测试一下麦克风音量

---

## 录制技巧

1. **提前录一遍不说话**，检查窗口切换是否流畅
2. **正式录的时候照着脚本念**，不用背
3. **鼠标移动要慢**，让观众能跟上
4. **如果有报错**，停下来重录这一段，后期可以剪
5. **不用一镜到底**，每段分开录，用剪映拼接
