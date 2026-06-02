# Knowledge Agent

[English](README.md)

基于 LangGraph 构建的个人知识库 Agent，参考了 [Letta](https://github.com/letta-ai/letta) 的三层记忆系统设计思路与 [Gemini Fullstack LangGraph Quickstart](https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart) 的 Agent 工作流架构。具备持续收集、整理、检索研究资料的能力，支持跨会话持久记忆。

## 功能特性

- **三层记忆系统** — Core Memory（常驻上下文 block）、Archival Memory（pgvector + BM25 混合检索）、Recall Memory（对话历史语义搜索）
- **混合检索管线** — 向量相似度 + BM25 关键词匹配 + cross-encoder 重排序（`BAAI/bge-reranker-v2-m3`）+ MMR 多样性过滤 + HyDE（假设答案嵌入）
- **选择性记忆捕获** — mem0 风格管线：LLM 判断 → 冲突解决 → 结构化 ADD/UPDATE/DELETE 操作，含会话上下文窗口
- **深度研究** — 多轮网页搜索，自动识别知识缺口并生成补充查询
- **文档摄入** — 支持 URL、PDF、纯文本导入，语义分块或固定大小分块
- **意图路由** — 闲聊 / 研究 / 回忆 / 编辑记忆 / 摄入文档
- **Core Memory 工具** — 自编辑 block，支持撤销历史和自动压缩
- **主动记忆推送** — 对话开始时自动推送相关记忆，无需用户询问
- **会话隔离** — Recall Memory 通过 `thread_id` 按会话隔离
- **全栈 UI** — React 前端，流式通信，文档库，对话历史侧边栏

## 演示

### 记忆检索

![记忆检索](images/memory_retrieved.png)

## 架构设计

### Agent 图结构

```
START → route_intent
  ├── "ingest" → ingest_document → respond → memory_pipeline → [consolidate?] → END
  └── (其他)   → recall_memory → evaluate_recall
                   ├── (记忆充足) → respond → memory_pipeline → [consolidate?] → END
                   └── (记忆不足) → generate_query → [web_research × N] → reflection
                         ├── (缺口) → 循环
                         └── (充足) → save_to_archival → respond → memory_pipeline → [consolidate?] → END
```

`evaluate_recall` 判断检索到的记忆是否足够回答问题——足够则跳过网页搜索。`memory_pipeline` 在每次回复后提取事实并进行冲突解决；`consolidate` 每 N 轮通过嵌入聚类合并重复条目。

### 检索管线

```
查询 → [改写] → [HyDE] → 搜索（向量 + BM25）→ 重排序（cross-encoder）→ [MMR 多样性] → 结果
```

- **查询改写** — 利用对话历史消除指代歧义
- **HyDE** — 生成假设答案，用其嵌入代替原始查询
- **MMR** — 平衡相关性与多样性

### 记忆管线

```
用户 + 助手消息
  → LLM 判断（值得记忆？）
  → 搜索已有记忆
  → 提取 ADD/UPDATE/DELETE 操作
  → 冲突解决（FACT_CONFLICT_PROMPT）
  → 执行操作
  → 存储会话上下文窗口（每 N 轮）
```

## 技术栈

| 组件 | 技术 |
|---|---|
| Agent 框架 | LangGraph StateGraph |
| LLM | 任意 OpenAI 兼容接口 |
| Embedding | DashScope `text-embedding-v3`（1024 维，带 LRU 缓存） |
| 向量存储 | PostgreSQL 16 + pgvector |
| 全文检索 | PostgreSQL tsvector + GIN |
| 重排序 | BAAI/bge-reranker-v2-m3（cross-encoder） |
| 搜索引擎 | Tavily API |
| 前端 | React 19 + Vite 6 + Tailwind CSS 4 + shadcn/ui |

## 快速开始

### 前置要求

- Python 3.11+
- Node.js 20+
- Docker & Docker Compose

### 1. 启动数据库

```bash
cd backend
docker compose up -d
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

```env
# LLM（OpenAI 兼容接口，支持 OpenAI / DeepSeek / MiMo 等）
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini

# Embedding（DashScope）
DASHSCOPE_API_KEY=your-dashscope-key
DASHSCOPE_EMBEDDING_MODEL=text-embedding-v3

# 网页搜索（Tavily）
TAVILY_API_KEY=your-tavily-key

# PostgreSQL
DATABASE_URL=postgresql://knowledge_agent:knowledge_agent@localhost:5432/knowledge_agent

# LangSmith（可选）
LANGSMITH_API_KEY=your-langsmith-key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=knowledge-agent
```

### 3. 安装与运行

```bash
# 后端
cd backend
pip install -e .
langgraph dev --no-reload

# 前端
cd frontend
npm install
npm run dev
```

后端运行于 `http://localhost:2024`，前端运行于 `http://localhost:5173`。

**CLI 模式**（无需前端）：`cd backend && python examples/cli_chat.py`

## 测试与评估

```bash
cd backend

# 单元测试
python -m pytest tests/ -q

# 混合检索召回率测试
python scripts/test_recall.py

# LoCoMo 评估（Recall@K，按问题类型分组）
python scripts/test_locomo.py --local data/locomo.json --limit 1        # 快速测试
python scripts/test_locomo.py --local data/locomo.json                   # 完整数据集
python scripts/test_locomo.py --local data/locomo.json --strategy all    # 对比所有策略
```

### 归档记忆管理

```bash
python scripts/manage_archival.py stats                                  # 各命名空间条目数
python scripts/manage_archival.py list --namespace research               # 列出条目
python scripts/manage_archival.py dedup --namespace research --dry-run    # 查找重复
python scripts/manage_archival.py smart-dedup --namespace research        # LLM 辅助去重
python scripts/manage_archival.py cleanup --namespace research --days 30  # 清理旧条目
python scripts/manage_archival.py consolidate --namespace research        # 完整整理
```

## 项目结构

```
knowledge-agent/
├── backend/
│   ├── src/agent/
│   │   ├── graph.py                  # LangGraph 定义
│   │   ├── state.py                  # AgentState TypedDict
│   │   ├── configuration.py          # 运行时配置
│   │   ├── prompts.py                # 提示词模板
│   │   ├── db.py                     # 数据库操作（archival + recall）
│   │   ├── memory/
│   │   │   ├── block.py              # Block 模型
│   │   │   ├── core_memory.py        # CoreMemory + 撤销历史
│   │   │   └── tools.py              # 记忆编辑 + 归档工具
│   │   ├── storage/
│   │   │   ├── embedding.py          # DashScope embedding + LRU 缓存
│   │   │   ├── reranker.py           # Cross-encoder 重排序
│   │   │   └── ingestion.py          # 文档摄入管线
│   │   └── nodes/
│   │       ├── memory_manager.py     # 意图路由 + 回忆 + 归档保存
│   │       ├── memory_pipeline.py    # 选择性捕获 + 整理
│   │       ├── researcher.py         # 网页研究 + 反思
│   │       └── responder.py          # 回答生成
│   ├── scripts/
│   │   ├── test_recall.py
│   │   ├── test_locomo.py
│   │   └── manage_archival.py
│   ├── tests/                        # 174 个单元测试
│   └── examples/cli_chat.py
└── frontend/src/
    ├── App.tsx
    └── components/
```

## 未来计划

- **时序查询支持** — 改进时间敏感问题的检索效果（如"周一发生了什么"、"上周的讨论"）。方案：摄入时提取时间元数据、时间感知查询扩展、归档搜索的日期范围过滤。
- **全命名空间编辑支持** — 当前仅 `manual` 命名空间的条目支持从文档库 UI 编辑，需扩展到 `ingested`、`research` 等其他命名空间。
- **版本 Diff 视图** — 支持两个版本之间的并排对比，高亮显示新增、删除、修改的文本，方便审查变更。
- **批量冲突处理** — 支持一次性批量批准或拒绝多条待审查的冲突记录，提供批量操作 UI。
- **Recall 与 Core Memory 版本追踪** — 将快照机制扩展到 Recall 记忆和 Core Memory 块，实现三层记忆全覆盖的回滚能力。
- **记忆分析仪表盘** — 可视化记忆质量指标：各命名空间条目数、冲突解决率、版本历史深度、来源可信度分布、存储增长趋势。
- **自动来源可信度评估** — 根据内容特征自动判断来源可靠性（如学术论文、官方文档 vs 博客文章），而非仅依赖 `metadata.source` 字段。
- **上下文工程优化**：
  - **Token 预算管理** — 对注入 LLM 的上下文（历史消息、检索结果、system prompt）做 token 计数与总量控制，防止超出模型上下文窗口。
  - **消息摘要压缩** — 对早期对话消息做摘要压缩，保留最近几轮原文，避免长对话中上下文丢失或 token 浪费。
  - **检索结果去重与冲突处理** — Archival 和 Recall 两路结果在注入前做内容去重，对矛盾信息做标注或取舍。
  - **上下文来源标注** — 为注入的检索结果添加结构化元数据（来源、时间、可信度），帮助 LLM 判断信息的可靠性和时效性。
  - **动态上下文分配** — 根据问题复杂度动态调整检索量和注入量，简单问题少注入、复杂问题多注入。

## 致谢

- **[Letta](https://github.com/letta-ai/letta)** — 三层记忆架构、Block 模型、自编辑记忆概念。
- **[Gemini Fullstack LangGraph Quickstart](https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart)** — 全栈 Agent 模式、研究循环、活动时间线 UI。

## 许可证

[MIT](LICENSE)
