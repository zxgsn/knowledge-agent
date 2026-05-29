# LoCoMo 召回率优化技术文档

> 记录 LoCoMo 基准测试中召回率从 R@10=26.6% 提升的完整技术分析与优化路径。

---

## 一、问题背景

在使用 LoCoMo（Long Conversation Memory）数据集对记忆管线进行基准测试时，基线（Baseline）策略的 Recall@K 指标表现极低：

```
Question Type            R@1     R@3     R@5    R@10
----------------------------------------------------
adversarial           22.9%   34.3%   41.4%   50.0%
event_summary         50.0%   50.0%   50.0%   50.0%
multi_session          0.0%    0.0%    0.0%    7.7%
single_session         3.1%    3.1%    6.2%   12.5%
temporal               0.0%    0.0%    0.0%    0.0%
----------------------------------------------------
Overall               11.7%   16.9%   20.8%   26.6%
```

以下是对每个根因的深入分析，以及对应的修复方案。

---

## 二、根因分析

### 根因 1：BM25 分支形同虚设（影响：高）

**现象：** 混合搜索公式为 `0.7 * cosine + 0.3 * ts_rank`，但 BM25 部分对最终得分几乎没有贡献。

**原因：** PostgreSQL 的 `plainto_tsquery('simple', query)` 会将查询中的**所有词元**（包括 "what"、"did"、"where" 等停用词）用 AND 逻辑组合。例如查询 `"What did Caroline research?"` 会被解析为：

```
'what' & 'did' & 'caroline' & 'research'
```

这意味着只有**同时包含这四个词**的文档才能匹配 BM25 条件。然而对话轮次通常是短句，如：

```
[Caroline] I applied to adoption agencies!
```

这句话不包含 "what"、"did"、"research"，因此 BM25 匹配完全失败。所有文档都是通过 `cosine > 0.2` 的兜底条件进入候选集的，BM25 的排序贡献为零。

**调试验证：** 实际查询 "What did Caroline research?" 的 top-10 结果中，`bm25_score` 全部为 0 或接近 0（0.001 量级），而 `cosine_score` 在 0.58–0.73 之间。混合得分实际上退化为纯向量搜索。

---

### 根因 2：BM25 与向量分数的量纲不匹配（影响：高）

**现象：** 即使 BM25 偶尔命中，其分数也无法有效参与排序。

**原因：** `ts_rank()` 返回的分数通常在 0.001–0.1 的范围内，而余弦相似度在 0.5–0.7 之间。在公式 `0.7 * cosine + 0.3 * ts_rank` 中：

```
假设 cosine = 0.65, ts_rank = 0.05
最终得分 = 0.7 * 0.65 + 0.3 * 0.05 = 0.455 + 0.015 = 0.470
```

BM25 仅贡献了 3% 的最终得分（0.015 / 0.470 ≈ 3.2%），远低于预期的 30% 权重。两个信号处于完全不同的数值尺度上，直接加权求和没有意义。

---

### 根因 3：原始对话轮次作为检索单元语义不足（影响：高）

**现象：** 问题 "What did Caroline research?"（答案：Adoption agencies）在 top-10 结果中找不到包含 "adoption agencies" 的轮次。

**原因：** 每条对话轮次以 `[speaker] utterance` 格式独立存储，例如：

```
[Caroline] I applied to adoption agencies! It's been a dream to have a family.
```

这条轮次的 embedding 是对话风格的，而问题是事实性问答风格。DashScope `text-embedding-v3` 在对话文本和问答文本之间的语义对齐能力有限，导致：

- 问题 embedding 与包含答案的轮次 embedding 余弦距离较远
- 反而是一些语义上更"泛"的轮次（如包含 "What" 的提问句）得分更高

**调试验证：** 查询 "What did Caroline research?" 的 top-10 结果全部是 Melanie 的提问句（"What happened that was so awesome?"、"What inspired it?"），因为这些句子在 embedding 空间中与问题更接近——它们都是疑问句。

---

### 根因 4：时间类问题结构性不可解（影响：中）

**现象：** temporal 类型召回率全部为 0%。

**原因：** LoCoMo 的时间类问题答案通常是绝对日期，如 `"7 May 2023"`，但对话中的时间表达是相对的：

```
Q: When did Caroline go to the LGBTQ support group?
A: 7 May 2023

对话原文: [Caroline] I went to a LGBTQ support group yesterday and it was so powerful.
```

"yesterday" 和 "7 May 2023" 在字面和语义上都无法匹配。这类问题需要**时间推理能力**（将对话的 session 日期与 "yesterday" 关联），纯检索系统无法解决。

---

### 根因 5：答案匹配逻辑过于严格（影响：中）

**现象：** 即使正确文档被检索到，匹配函数也可能判定为"未命中"。

**原因：** `answer_in_results()` 使用两种匹配方式：

1. **子串匹配**：要求答案完整出现在文档中（大小写不敏感）
2. **词元重叠**：答案的词元 >50% 出现在文档中

问题场景：
- 答案 `"self-care is important"` vs 文档 `"She realized that taking care of herself matters"` — 子串不匹配，词元重叠不足 50%
- 答案 `"mental health"` 仅 2 个词元，匹配 1 个即达 50% — 过于宽松，容易误判

---

### 根因 6：对抗性问题占比过高（影响：中）

**现象：** adversarial 类型占全部问题的 42%（841/1986），且 R@10 仅 50%。

**原因：** 对抗性问题测试的是系统区分真实信息与虚构上下文的能力。例如：

```
Q: What did the charity race raise awareness for?
A: mental health
```

这类问题需要理解对话的完整上下文，而非简单的关键词匹配。它要求检索系统不仅找到相关段落，还要正确理解事件的因果关系。

---

## 三、优化方案

### 修复 1：激活 BM25 — 使用 `websearch_to_tsquery`

**改动：** 将 `plainto_tsquery('simple', query)` 替换为 `websearch_to_tsquery('simple', query)`。

**原理：** `websearch_to_tsquery` 使用 OR 逻辑连接词元（空格分隔的词默认为 OR），并支持引号短语。查询 `"What did Caroline research?"` 被解析为：

```
'what' | 'did' | 'caroline' | 'research'
```

任何包含这四个词中**任意一个**的文档都能获得 BM25 分数。这使得包含 "caroline" 或 "research" 的对话轮次能够被 BM25 信号捕获。

**影响文件：**
- `backend/scripts/test_locomo.py` — `search_archival()` 函数
- `backend/src/agent/storage/archival.py` — `search()` 方法

---

### 修复 2：BM25 分数归一化

**改动：** 将 `ts_rank(...)` 包裹在 `LEAST(1, ts_rank(...) * 10)` 中。

**原理：** 乘以 10 的系数将 ts_rank 的典型值（0.01–0.1）提升到 0.1–1.0 范围，与余弦相似度处于同一量纲。`LEAST(1, ...)` 确保不超过 1.0。

修复后的混合公式：

```sql
0.7 * cosine_similarity
  + 0.3 * LEAST(1, ts_rank(content_tsv, websearch_to_tsquery('simple', query)) * 10)
```

**效果：** BM25 信号真正参与排序，而非被余弦分数淹没。

---

### 修复 3：简化 WHERE 过滤条件

**改动：** 移除 WHERE 子句中的 BM25 匹配条件，仅保留余弦阈值：

```sql
-- 修复前
WHERE namespace = %s
  AND (content_tsv @@ plainto_tsquery('simple', %s)
       OR 1 - (embedding <=> %s::vector) > 0.2)

-- 修复后
WHERE namespace = %s
  AND 1 - (embedding <=> %s::vector) > 0.15
```

**原因：** 原来的 BM25 WHERE 条件（AND 逻辑）几乎从不命中，是一个死代码路径。移除后逻辑更清晰，阈值从 0.2 降至 0.15 以保留更多候选文档。

---

### 修复 4：会话级上下文窗口（新增策略）

**改动：** 新增 `ingest_session_context()` 函数，存储重叠的多轮对话窗口而非单轮。

**原理：** 单轮对话 `[Caroline] I applied to adoption agencies!` 缺乏上下文。将 5 轮对话合并为一个检索单元：

```
[Session 3] Caroline: Guess what I did this week? I took the first step towards becoming a mom.
Melanie: Wow, that's amazing! What did you do?
Caroline: I applied to adoption agencies! It's been a dream to have a family.
Melanie: That's so exciting! What kind of agencies?
Caroline: Ones that support LGBTQ+ individuals.
```

这个窗口的 embedding 包含更丰富的语义信息，更容易被事实性问题检索到。

**参数：** `window_size=5`, `stride=2`（每 2 轮滑动一次，窗口大小 5 轮）。

---

### 修复 5：LLM 答案判定器（可选）

**改动：** 新增 `--llm-judge` 标志，使用 LLM 判断检索结果是否包含预期答案。

**原理：** 子串匹配无法处理释义（paraphrase）。LLM 判定器将问题、预期答案和 top-K 检索结果一起发给 LLM，由 LLM 判断语义上是否匹配。例如：

```
问题: What did Melanie realize after the charity race?
预期答案: self-care is important
检索结果: [Melanie] After the run, I figured out that looking after myself has to come first.

LLM 判断: yes (语义等价)
```

**代价：** 每个问题需要一次 LLM 调用，评估速度大幅下降。默认关闭，仅在需要精确评估时启用。

---

## 四、优化前后对比

| 指标 | 优化前 | 预期优化后 |
|------|--------|-----------|
| BM25 实际贡献 | ≈ 0% | 15–30% |
| 检索单元 | 单轮对话 | 5轮上下文窗口 |
| 答案匹配 | 子串 + 词元 | 子串 + 词元 + LLM（可选） |
| temporal 召回 | 0% | 取决于数据（结构性难题） |
| 单轮语义 | 对话风格 | 问答风格（提取后） |

---

## 五、已知局限

1. **时间推理无法通过检索解决：** "yesterday" → "7 May 2023" 需要时间解析能力，这不是检索系统的职责。需要在管线外增加时间归一化层。

2. **对抗性问题的本质难度：** 区分真实信息与虚构上下文需要深层语义理解，超出当前 embedding 模型的能力范围。

3. **event_summary 问题需要多文档综合：** 这类问题的答案分散在多个对话轮次中，需要跨文档推理能力。

4. **DashScope embedding 的领域适配性：** `text-embedding-v3` 通用模型在对话文本上的表现可能不如专门针对对话微调的模型。

---

## 六、相关文件

| 文件 | 改动内容 |
|------|---------|
| `backend/scripts/test_locomo.py` | BM25 修复、会话窗口策略、LLM 判定器、CLI 参数 |
| `backend/src/agent/storage/archival.py` | 生产环境 BM25 修复 |
| `backend/data/locomo.json` | LoCoMo 数据集（.gitignore 排除） |

---

## 七、验证命令

```bash
cd backend

# 基线测试（快速，仅 BM25 修复效果）
python scripts/test_locomo.py --local data/locomo.json --limit 1

# 会话上下文窗口策略
python scripts/test_locomo.py --local data/locomo.json --limit 1 --strategy context

# 全部 4 种策略对比
python scripts/test_locomo.py --local data/locomo.json --limit 1 --strategy all

# 启用 LLM 答案判定（慢，但更准确）
python scripts/test_locomo.py --local data/locomo.json --limit 1 --strategy all --llm-judge
```
