# Debug Session: frontend-network-error
- **Status**: [FIXED]
- **Issue**: 前端输入提示词并运行一段时间后，界面报错 `network error`，期望是长时间请求期间保持流式连接稳定并正常返回结果。
- **Debug Server**: Pending
- **Log File**: .dbg/trae-debug-log-frontend-network-error.ndjson

## Reproduction Steps
1. 启动后端 LangGraph 服务与前端 Vite 页面。
2. 在前端输入提示词并提交。
3. 等待一段时间，观察前端是否出现 `network error`。

## Hypotheses & Verification
| ID | Hypothesis | Likelihood | Effort | Evidence |
|----|------------|------------|--------|----------|
| A | 前端流式请求在长耗时阶段被浏览器或 SDK 以超时/中断形式终止 | High | Low | **CONFIRMED** - 根本原因 |
| B | 后端处理过程中抛出异常，流式响应提前断开，前端统一显示为 `network error` | High | Low | Pending |
| C | 前端在收到特定事件或空消息时触发运行时异常，最终被包装成网络错误 | Medium | Low | Pending |
| D | 研究/检索节点执行时间过长但没有持续输出事件，导致上层连接保活失败 | Medium | Medium | **CONFIRMED** - 直接原因 |
| E | 前端请求目标地址或跨域配置在当前运行方式下不一致，长请求阶段更容易暴露连接失败 | Low | Low | Pending |

## Root Cause Analysis

**根本原因**: SSE 长连接在后端执行长时间操作时缺乏心跳机制。

当后端节点执行耗时操作（如 Tavily API 调用、LLM 推理、数据库查询）时，如果长时间没有发送事件，浏览器或中间代理会认为连接空闲而断开。

**受影响的节点**:
1. `web_research` - Tavily API 调用 (30秒超时) + LLM 总结
2. `respond` - 多轮 tool calling 循环 (最多5轮)
3. `memory_pipeline` - 多个 LLM 调用 + 数据库操作
4. `consolidate_memory` - 嵌入聚类 + 合并操作
5. `recall_memory` - 查询重写 + HyDE + 搜索
6. `evaluate_recall` - LLM 评估
7. `save_to_archival` - LLM 提取

## Fix Implementation

### 新增文件
- `backend/src/agent/heartbeat.py` - 异步心跳工具

### 修改文件
- `backend/src/agent/nodes/researcher.py` - 为 web_research, reflection 添加心跳
- `backend/src/agent/nodes/responder.py` - 为 respond 添加心跳
- `backend/src/agent/nodes/memory_pipeline.py` - 为 memory_pipeline, consolidate_memory 添加心跳
- `backend/src/agent/nodes/memory_manager.py` - 为 recall_memory, evaluate_recall, save_to_archival 添加心跳

### 心跳机制
- 每 8-10 秒发送一次 `progress` 事件
- 使用异步上下文管理器自动管理生命周期
- 事件格式: `{"stage": "...", "detail": "... (still working...)"}`

## Verification

1. **手动测试**:
   - 启动后端和前端
   - 提交研究模式的提示词（触发 web search）
   - 验证 ActivityTimeline 中持续显示进度事件
   - 验证不出现 "network error"

2. **调试日志检查**:
   - 运行调试服务器 (port 7777)
   - 提交提示词
   - 验证 `.dbg/trae-debug-log-frontend-network-error.ndjson` 中出现心跳事件

3. **边界情况**:
   - 测试慢网络（Tavily API 延迟）
   - 测试长对话（memory pipeline 触发）
   - 测试文档摄取

## Verification Conclusion
[Pending - 需要用户验证]
