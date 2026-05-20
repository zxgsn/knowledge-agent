# Knowledge Agent

[English](README.md)

基于 LangGraph 构建的个人知识库 Agent，参考了 [Letta](https://github.com/letta-ai/letta) 的三层记忆系统设计思路与 [Gemini Fullstack LangGraph Quickstart](https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart) 的 Agent 工作流架构。具备持续收集、整理、检索研究资料的能力，支持跨会话持久记忆。

## 功能特性

- **三层记忆系统** — Core Memory（常驻上下文）、Archival Memory（向量长期存储）、Recall Memory（对话历史）
- **混合检索** — Archival Memory 检索结合向量语义相似度（DashScope embedding）与 BM25 关键词匹配（PostgreSQL tsvector + GIN 索引），提升召回率
- **Core Memory 自动压缩** — block 接近字符上限时，LLM 自动提炼压缩内容，保留关键信息
- **深度研究** — 多轮网页搜索，自动识别知识缺口并生成补充查询
- **文档摄入** — 支持 URL、PDF、纯文本导入，自动分块与向量化
- **意图路由** — LLM 自动分类：闲聊 / 研究 / 回忆 / 编辑记忆 / 摄入文档
- **闲聊检索知识库** — 闲聊模式自动搜索 Archival Memory，将相关知识注入上下文
- **持久化知识库** — 研究发现存储于 PostgreSQL + pgvector，跨会话可检索
- **全栈 UI** — React 前端，实时展示 Agent 工作进度

## 架构设计

### 三层记忆系统

```
┌─────────────────────────────────────────────────────┐
│                    Agent State                       │
│                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ Core Memory  │  │ Archival Mem  │  │ Recall Mem │ │
│  │ (始终在上下文)│  │ (语义长期存储) │  │ (对话历史) │ │
│  │              │  │              │  │            │ │
│  │ persona      │  │ pgvector     │  │ pgvector   │ │
│  │ human        │  │ 向量检索      │  │ 语义检索    │ │
│  │ knowledge_   │  │ + 元数据过滤  │  │            │ │
│  │   focus      │  │              │  │            │ │
│  └─────────────┘  └──────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────┘
```

- **Core Memory** — 常驻 system prompt 的可编辑 block，agent 可自我编辑；block 接近上限时自动 LLM 压缩
- **Archival Memory** — pgvector 语义存储，持久化研究发现，支持跨会话检索；混合检索（向量余弦相似度 70% + BM25 关键词匹配 30%）
- **Recall Memory** — 对话历史的语义搜索（表结构已就绪，集成待完成）

### Agent 图结构

```
START → route_intent
  ├── "chat"        → recall_memory → respond → END
  ├── "research"    → generate_query → [web_research × N] → reflection
  │                   ├── (知识缺口) → [web_research] → reflection (循环)
  │                   └── (信息充足) → save_to_archival → respond → END
  ├── "recall"      → recall_memory → respond → END
  ├── "memory_edit" → recall_memory → respond → END
  └── "ingest"      → ingest_document → save_to_archival → respond → END
```

所有非研究路径都会先经过 `recall_memory`，确保 Agent 回答时始终有相关知识上下文。

### 文档摄入管道

```
用户输入 URL / 纯文本 / 文件路径
  ↓
ingest_document 节点
  ├── URL → httpx 获取 → BeautifulSoup 提取正文
  ├── 纯文本 → 直接处理
  └── 文件路径 → 读取文件内容
  ↓
chunk_text() 分块
  ├── 按段落边界分割
  ├── 段落内按句子边界二次分割
  └── 支持 overlap 保持上下文连贯
  ↓
DashScope Embedding 批量向量化
  ↓
ArchivalMemory.put_batch() 写入 pgvector
```

## 技术栈

| 组件 | 技术 | 说明 |
|---|---|---|
| Agent 框架 | LangGraph StateGraph | 状态图驱动的 Agent 工作流 |
| LLM | 任意 OpenAI 兼容接口 | 通过 `.env` 配置 |
| Embedding | DashScope `text-embedding-v3` | 1024 维向量 |
| 向量存储 | PostgreSQL 16 + pgvector | Docker 部署，余弦相似度检索 |
| 全文检索 | PostgreSQL tsvector + GIN | BM25 关键词匹配，与向量混合检索 |
| 搜索引擎 | Tavily API | 高级搜索模式 |
| 文档处理 | pypdf + BeautifulSoup + httpx | PDF / 网页 / 文本提取 |
| 前端 | React 19 + Vite 6 + Tailwind CSS 4 | shadcn/ui 组件库 |

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

启动 PostgreSQL 16 + pgvector 扩展，默认端口 5432。

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 填入 API key：

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

# LangSmith（可选，用于链路追踪与可观测性）
LANGSMITH_API_KEY=your-langsmith-key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=knowledge-agent
```

### 3. 安装后端依赖

```bash
cd backend
pip install -e .
```

### 4. 安装前端依赖

```bash
cd frontend
npm install
```

### 5. 启动服务

**后端（LangGraph dev server）：**

```bash
cd backend
langgraph dev
```

默认运行在 `http://localhost:2024`。

**前端（Vite dev server）：**

```bash
cd frontend
npm run dev
```

默认运行在 `http://localhost:5173`。

**CLI 模式（无需前端）：**

```bash
cd backend
python examples/cli_chat.py
```

### 6. LangSmith 链路追踪（可选）

[LangSmith](https://smith.langchain.com/) 提供 Agent 的链路追踪与可观测性。启用步骤：

1. 在 [smith.langchain.com](https://smith.langchain.com/) 注册并获取 API key
2. 在 `.env` 中设置环境变量：
   ```env
   LANGSMITH_API_KEY=lsv2_pt_xxxxx
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_PROJECT=knowledge-agent
   ```
3. 重启 `langgraph dev` — 所有 LLM 调用、工具调用和图执行都会自动被追踪

在 `https://smith.langchain.com/` 的项目页面查看 trace。每次运行展示完整的执行图，包含各节点的耗时、输入输出和 token 用量。

## 项目结构

```
knowledge-agent/
├── backend/
│   ├── pyproject.toml
│   ├── docker-compose.yml            # PostgreSQL + pgvector
│   ├── .env.example
│   ├── langgraph.json
│   ├── examples/
│   │   └── cli_chat.py               # CLI 入口
│   ├── scripts/
│   │   └── test_recall.py            # 混合检索召回率测试
│   └── src/agent/
│       ├── graph.py                  # 主图定义
│       ├── state.py                  # AgentState TypedDict
│       ├── configuration.py          # 运行时配置
│       ├── prompts.py                # 提示词模板
│       ├── memory/
│       │   ├── block.py              # Block 模型（参考 Letta）
│       │   ├── core_memory.py        # CoreMemory + compile()
│       │   └── tools.py              # 记忆编辑 + 归档工具
│       ├── storage/
│       │   ├── embedding.py          # DashScope embedding 封装
│       │   ├── archival.py           # ArchivalMemory (pgvector)
│       │   ├── recall.py             # RecallMemory
│       │   └── ingestion.py          # 文档摄入管道
│       └── nodes/
│           ├── memory_manager.py     # 意图路由 + 归档保存
│           ├── researcher.py         # 搜索 + 研究 + 反思
│           └── responder.py          # 回答生成
└── frontend/
    └── src/
        ├── App.tsx                   # 主应用，LangGraph 流式通信
        └── components/
            ├── InputForm.tsx         # 模式选择器
            ├── WelcomeScreen.tsx
            ├── ChatMessagesView.tsx
            └── ActivityTimeline.tsx
```

## CLI 命令

| 命令 | 说明 |
|---|---|
| `/memory` | 查看 Core Memory blocks |
| `/search <query>` | 搜索 Archival Memory |
| `/ingest <url_or_path>` | 摄入文档（URL 或文件路径） |
| `/quit` | 退出 |

## 致谢

本项目的设计思路参考了以下开源项目：

- **[Letta](https://github.com/letta-ai/letta)**（前身 MemGPT）— 三层记忆架构（Core / Archival / Recall）、Block 记忆模型、以及 Agent 自编辑记忆的概念均参考自 Letta。本项目的实现是独立的，并针对 LangGraph 框架进行了简化。
- **[Gemini Fullstack LangGraph Quickstart](https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart)** — 全栈 Agent 模式（LangGraph 后端 + React 流式前端）、带反思的研究循环、以及活动时间线 UI 均参考自 Google Gemini 的这个快速入门项目。

## 许可证

[MIT](LICENSE)
