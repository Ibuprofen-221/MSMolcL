# 代码库问题分析报告

> 分析日期：2026-05-06 | 范围：前端 + 后端全栈

---

## 一、急需解决 (Urgent)

### 1. 安全漏洞

#### 1.1 JWT 密钥硬编码默认值
- **位置**: `backend/core/config.py:25`
- **问题**: `jwt_secret_key` 默认值为 `"please-change-this-secret-in-production"`，生产环境若未设置环境变量，任何人均可伪造 JWT Token。
- **影响**: 认证系统完全失效。
- **建议**: 启动时检测是否为默认值，若是则拒绝启动并打印错误日志。

#### 1.2 登录/注册接口无限流保护
- **位置**: `backend/api/auth.py:16-71`
- **问题**: `/api/register` 和 `/api/login` 未使用 `@limiter.limit` 装饰器，可被暴力破解。
- **影响**: 用户密码可被字典攻击枚举。
- **建议**: 对登录/注册接口添加 stricter 限流规则（如每分钟 5 次），并添加登录失败递增延迟。

#### 1.3 Token 存储在 localStorage
- **位置**: `frontend/src/utils/storage.js:93-97`
- **问题**: JWT Token 存储在 `localStorage`，任何 XSS 攻击可读取并窃取 token。
- **影响**: 账户接管风险。
- **建议**: 改用 httpOnly + Secure + SameSite 的 Cookie 存储 token，或至少使用 sessionStorage + 较短的过期时间。

#### 1.4 无密码强度验证
- **位置**: `backend/api/auth.py:17-21`
- **问题**: 注册时仅检查密码非空，无长度/复杂度要求。
- **影响**: 弱密码易被破解。
- **建议**: 最小长度 8 位，要求包含字母+数字。

#### 1.5 `torch.load(weights_only=False)` 任意代码执行
- **位置**: `backend/services/retrieve.py:658`, `backend/services/retrieve_advanced.py:106`, `backend/services/model/predict.py:223`
- **问题**: `torch.load(model_weight_path, weights_only=False)` 允许任意 pickle 反序列化。若攻击者可替换模型权重文件，即可远程执行任意代码 (RCE)。
- **影响**: 服务器完全沦陷。
- **建议**: 尽可能使用 `weights_only=True`；如必须使用完整模型，需对模型文件做哈希校验 + 数字签名验证。

#### 1.6 SMILES 可视化接口无限流无认证
- **位置**: `backend/api/smiles_visualization.py:32-57`
- **问题**: 接口无需认证、无限流，可提交任意 SMILES 字符串触发 RDKit 渲染，消耗 CPU/GPU 资源。
- **影响**: 匿名 DoS 攻击向量。
- **建议**: 添加认证依赖 + 限流装饰器。

#### 1.7 源码泄露 GitHub Token 与内部 URL
- **位置**: `backend/update_url.py:5-8`
- **问题**: 文件包含硬编码的 Gist ID 和 Token 占位符，以及 seetacloud.com 内部基础设施 URL。
- **影响**: 泄露内部部署拓扑，若 Token 被误填入将导致数据泄露。
- **建议**: 从版本控制中移除此文件，加入 `.gitignore`。

#### 1.8 Task ID 可猜测导致越权访问
- **位置**: `backend/api/statas.py:54-56`, `backend/api/retrieve.py:56-61`
- **问题**: Task ID 由 `uuid4().hex` 生成（32位hex），虽难猜测但仍为不透明标识符。AdvancedSearch 页面允许任意输入 task_id 加载他人任务数据，仅靠 `_ensure_task_belongs_to_user` 做归属校验。
- **影响**: 若用户泄露 task_id，他人可访问其数据。
- **建议**: 在 `_ensure_task_belongs_to_user` 基础上，对敏感接口增加二次用户身份绑定检查。

---

### 2. 并发与线程安全

#### 2.1 模块全局变量竞态条件
- **位置**: `backend/services/file_preprocess.py:20-24`（`SPECTRUM_OUTPUT_DIR`, `VALID_PAIRS_MGF` 等）
- **问题**: `set_output_base()` 修改模块级全局变量，不同请求同时调用 `preprocess_main()` 会互相覆盖输出路径，导致数据错乱。
- **影响**: 并发上传时，用户 A 的数据可能写入用户 B 的任务目录，或产生不完整的输出文件。
- **建议**: 移除全局变量，将输出路径作为参数传入 `SpectrumNormalizer` / `FragTreeParser` 构造函数。

#### 2.2 BATCH_JOB_GROUPS 无过期清理
- **位置**: `backend/api/retrieve.py:28`
- **问题**: `BATCH_JOB_GROUPS` 是模块级字典，存储批次任务组信息，永不过期清理。每个多文件检索任务会新增一条记录。
- **影响**: 长期运行后内存泄漏。
- **建议**: 添加定时清理协程或使用 TTL 缓存（如 `cachetools.TTLCache`）。

#### 2.3 内存缓存无上限无过期
- **位置**: `backend/core/memory_store.py:1-9`
- **问题**: `processed_cache` 和 `custom_lib_cache` 是全局字典，`custom_lib_cache` 以文件名为 key 存储 SMILES 列表，永不清除。
- **影响**: 用户反复上传自定义库文件，内存持续增长。
- **建议**: 使用 LRU 缓存或基于 task 生命周期的缓存策略。

#### 2.4 自定义异常继承自 BaseException
- **位置**: `backend/core/exceptions.py:1,5,9,13`
- **问题**: `FileMissingError`, `FileFormatError`, `FileSizeError` 继承自 `BaseException` 而非 `Exception`。`BaseException` 是 `SystemExit`/`KeyboardInterrupt` 的基类，自定义业务异常不应继承它。
- **影响**: 这些异常不会被 `except Exception:` 块捕获，可能导致未预期的异常传播行为。
- **建议**: 改为继承 `Exception`。

#### 2.5 `history_store.py` 无文件锁保护
- **位置**: `backend/services/history_store.py:83-88,120-147`
- **问题**: `upsert_task_record` / `delete_task_record` 的读→修改→原子写流程无文件级锁。同一用户的并发请求会导致写丢失 (lost update)。
- **影响**: 历史记录更新丢失，数据不一致。
- **建议**: 使用 `fcntl.flock` 或 `threading.Lock` + 用户级键保护原子写入。

#### 2.6 GPU Worker 进程静默崩溃检测滞后
- **位置**: `backend/services/retrieve_runtime.py:306-314`
- **问题**: `_sync_worker_state()` 仅在 `submit()` 和 `get_job()` 时被调用。若无新请求，Worker 崩溃后系统毫无感知。
- **影响**: 已提交的作业永久处于 pending/running 状态。
- **建议**: 添加后台心跳线程，每 5-10 秒检测 Worker 存活状态。

---

### 3. 阻塞与性能

#### 3.1 同步预处理阻塞事件循环
- **位置**: `backend/api/file_upload.py:238`, `backend/services/sirius_batch_service.py:214`
- **问题**: `preprocess_main()` 涉及大量磁盘 I/O、JSON 解析、NumPy 数组操作，在 async 请求处理函数中同步调用，阻塞整个 event loop。
- **影响**: 预处理期间所有其他 API 请求被阻塞（包括 health check），服务表现为假死。
- **建议**: 使用 `asyncio.to_thread()` 或 `run_in_executor()` 将预处理移到线程池执行。

#### 3.2 聚合操作阻塞状态轮询请求
- **位置**: `backend/services/sirius_batch_service.py:386`
- **问题**: `get_mgf_only_batch_status()` 在请求处理中同步调用 `_run_aggregate_if_needed()`，该函数执行大量文件读写、预处理和 JSON 序列化。
- **影响**: 前端轮询请求会因聚合操作而阻塞 30+ 秒，超过浏览器/代理的默认超时时间。
- **建议**: 聚合操作应在后台线程或 Celery 任务中异步执行，状态轮询仅返回聚合进度。

#### 3.3 无 Job 执行超时
- **位置**: `backend/services/retrieve_runtime.py:82-106`
- **问题**: `execute_retrieve_job()` 没有超时限制，若 GPU 推理卡死，Worker 进程永久阻塞。
- **影响**: 该 ion_mode 的所有后续作业永远无法执行。
- **建议**: 使用 `multiprocessing.Process` 执行每个 job 并设置 `join(timeout=N)`。

---

### 4. 用户数据管理

#### 4.1 无用户存储配额
- **位置**: 全局文件系统
- **问题**: 用户可以无限次上传文件创建任务，无磁盘配额限制。
- **影响**: 恶意用户可填满服务器磁盘。
- **建议**: 按用户配置文件大小上限，或限制每用户最大任务数。

#### 4.2 任务文件永不自动清理
- **位置**: `backend/core/config.py:73`（`retrieve_task_keep_seconds` 仅影响内存中的 job 状态）
- **问题**: 用户创建的 task 目录（含 MGF/JSON/statas 等文件）永不被自动删除。
- **影响**: 磁盘空间持续增长。
- **建议**: 添加定时任务清理超过 N 天的旧 task 目录。

#### 4.3 历史记录单文件写入
- **位置**: `backend/services/history_store.py:83-88`
- **问题**: 所有用户历史记录存储为单个 JSON 文件，每次更新需要读→修改→原子写整个文件。用户任务多时性能恶化。
- **影响**: 数百条记录时每次更新都要读写整个文件。
- **建议**: 超过一定阈值后改为 SQLite 存储，或拆分按月份归档。

---

### 5. 前端问题

#### 5.1 轮询无退避策略
- **位置**: `frontend/src/components/ChemSearch.vue:392-425` (upload polling), `:606-646` (retrieve polling)
- **问题**: 每 2 秒轮询一次，超时最长 1 小时。一个检索任务最多产生 1800 次 HTTP 请求。无 exponential backoff。
- **影响**: 服务端压力大，移动网络下耗电耗流量。
- **建议**: 前 1 分钟每 2s 轮询，之后逐步增加到 5s、10s、30s。

#### 5.2 组件卸载时未取消网络请求
- **位置**: `frontend/src/components/ChemSearch.vue:361-365`, `:652`
- **问题**: `onBeforeUnmount` 设置了 `stopRetrievePolling = true`，但正在进行的 HTTP 请求不会被 abort。组件卸载后请求响应仍会尝试更新已销毁的响应式状态。
- **影响**: 潜在的 memory leak 和控制台警告 "response error on unmounted component"。
- **建议**: 使用 `AbortController` 在卸载时取消进行中的请求。

#### 5.3 大型数据结构写入 sessionStorage
- **位置**: `frontend/src/components/ChemSearch.vue:486,625` 等处
- **问题**: statas 完整数据（含所有谱图信息）序列化后写入 sessionStorage（通常限制 5-10MB）。大数据集可能超限。
- **影响**: sessionStorage 满后写入静默失败（不抛异常），用户刷新后丢失状态。
- **建议**: 仅持久化关键元数据（taskId, file paths），大数据通过 API 重新获取。

#### 5.4 状态提早重置导致数据丢失
- **位置**: `frontend/src/components/ChemSearch.vue:461-462,654`, `AdvancedSearch.vue:310,390`
- **问题**: 在 API 调用完成前即设置 `retrieveSummary.value = null` / `chooseData.value = null`。若请求失败，先前的结果永久丢失无法恢复。
- **影响**: 重试失败后用户看到空白结果页。
- **建议**: 在 API 成功返回后再清除旧状态。

#### 5.5 重复错误提示
- **位置**: `frontend/src/components/ChemSearch.vue:219-221`, `AdvancedSearch.vue:173-176`
- **问题**: `setGlobalError` 同时调用 `ElMessage.error()` 和设置 `globalError.value`，导致同一条错误同时弹出 toast 通知和固定位置的 alert 横幅。
- **影响**: 用户体验差，双重噪音。
- **建议**: 选择一种错误展示方式（toast 或 alert，非两者同时）。

#### 5.6 History 页面每条记录发起 2 次 API 调用
- **位置**: `frontend/src/views/HistoryRecords.vue:103`
- **问题**: `loadStatusCount` 对每条记录分别调用 `fetchStatas`（normal + advanced 各一次）。30 条记录 = 60 次 API 调用。
- **影响**: 页面加载极慢，服务端压力大。
- **建议**: 合并为单次批量查询 API，或添加分页限制每页条数。

#### 5.7 错误信息直接暴露后端细节
- **位置**: `frontend/src/components/ChemSearch.vue:161`
- **问题**: `error?.response?.data?.detail || error?.response?.data?.message || error?.message` 直接将后端错误消息展示给用户。
- **影响**: 可能泄露内部路径、数据库结构等信息。
- **建议**: 前端做错误消息白名单映射，不直接透传后端异常信息。

---

## 二、后期升级 (Future Upgrade)

### 1. 架构与代码质量

#### 1.1 ChemSearch.vue 巨型组件
- **位置**: `frontend/src/components/ChemSearch.vue` (1022 行)
- **问题**: 单个组件承担 upload、file preview、candidate select、retrieval、result display 全部逻辑。无子组件拆分。
- **建议**: 拆分为 UploadStep.vue、FileCardsStep.vue、CandidateStep.vue、ResultStep.vue，共享逻辑提取为 composables。

#### 1.2 ChemSearch 与 AdvancedSearch 大量重复
- **位置**: `ChemSearch.vue` + `AdvancedSearch.vue`
- **问题**: 两组件共享 ~80% 的模板和逻辑（文件卡片展示、谱图选择、Plotly 渲染、结果表格），但完全独立编写。
- **建议**: 提取共享子组件：SpectrumFileList.vue、SpectrumPlotPanel.vue、ResultTablePanel.vue。

#### 1.3 无 TypeScript
- **位置**: 整个 frontend 目录
- **问题**: 纯 JavaScript，无类型检查。API 响应结构、session 数据格式、组件 props 均无类型约束。
- **建议**: 渐进迁移到 TypeScript，至少对 API 层和 utils 层加类型。

#### 1.4 前端无路由级代码分割
- **位置**: `frontend/src/router/index.js:3-8`
- **问题**: 所有路由组件同步 import，首屏加载所有页面代码。
- **建议**: 使用 `() => import(...)` 动态导入实现路由级 lazy loading。

#### 1.5 两个检索运行时代码近乎完全重复
- **位置**: `backend/services/retrieve_runtime.py` 与 `backend/services/retrieve_advanced_runtime.py`
- **问题**: 两个文件结构几乎完全相同（仅类名和 config key 不同），约 350 行重复代码。
- **建议**: 提取公共基类 `BaseRetrieveRuntime`，两个子类仅覆盖差异部分。

#### 1.6 前端存在不可达死代码
- **位置**: `frontend/src/components/HelloWorld.vue`（Vite 模板残留），`frontend/src/views/SpectraVisualizer.vue`（未注册路由）
- **问题**: 死代码增加维护负担，且 SpectraVisualizer 引用不存在的前端路由仍出现在代码树中。
- **建议**: 删除未使用的文件。

#### 1.7 Plotly 全量引入
- **位置**: `frontend/src/components/ChemSearch.vue:3`
- **问题**: `plotly.js-dist-min` 体积约 3MB，但仅使用了基础的 scatter 图表。
- **建议**: 改用 `plotly.js-basic-dist` 或按需引入 plotly.js 模块，减少约 2MB。

---

### 2. 后端架构

#### 2.1 无结构化日志
- **位置**: 整个 backend
- **问题**: 使用 `print()` 而非 logging 模块，无日志级别、无请求 ID 追踪。
- **建议**: 引入 `loguru` 或标准 `logging`，配置请求 ID 中间件，记录关键操作日志。

#### 2.2 无配置校验
- **位置**: `backend/core/config.py`
- **问题**: 使用模块级变量方式配置，无启动时校验。模型文件缺失、数据库目录不存在仅在运行时才报错。
- **建议**: 使用 Pydantic Settings 进行配置建模，启动时校验所有路径和依赖。

#### 2.3 数据库无连接池配置
- **位置**: `backend/core/db.py:6`
- **问题**: SQLite 使用默认连接参数，`check_same_thread=False` 允许多线程但无池大小控制。
- **建议**: 配置 `pool_size` 和 `max_overflow`，对 SQLite 考虑升级到 PostgreSQL 以支持更好的并发。

#### 2.4 无分布式 GPU Worker
- **位置**: `backend/services/retrieve_runtime.py`
- **问题**: GPU Worker 与 Web 服务器在同一进程树（`multiprocessing.Process`），无法扩展到多机。
- **建议**: 将 GPU Worker 抽象为独立服务（通过 Celery/RabbitMQ 或 gRPC 通信），支持水平扩展。

#### 2.5 History API 无分页
- **位置**: `backend/api/history.py:20-22`
- **问题**: `GET /api/history` 返回全部记录，无分页参数。
- **影响**: 用户积累数百条任务后响应体过大。
- **建议**: 添加 `page`/`page_size` 查询参数。

---

### 3. 用户体验

#### 3.1 无 WebSocket 实时推送
- **位置**: 前端轮询 → 后端状态查询
- **问题**: 检索/上传状态通过 2s 间隔 HTTP 轮询获取，延迟高、服务端压力大。
- **建议**: 引入 WebSocket 或 SSE，状态变更时主动推送给前端。

#### 3.2 无离线/断网提示
- **位置**: 前端全局
- **问题**: 网络断开时前端无感知，HTTP 请求静默失败，用户体验差。
- **建议**: 添加 `navigator.onLine` 监听和网络状态指示器。

#### 3.3 无加载骨架屏
- **位置**: 前端各步骤
- **问题**: 数据加载时使用 `v-loading` 或 `loading` 变量控制，无 skeleton/placeholder UI。
- **建议**: 添加 Element Plus `<el-skeleton>` 组件提升感知速度。

#### 3.4 移动端适配不足
- **位置**: `frontend/src/components/ChemSearch.vue:1016-1020`
- **问题**: 仅一个 `@media (max-width: 1100px)` 断点，平板竖屏和手机横屏均未适配。
- **建议**: 增加 768px / 480px 断点的响应式布局。

#### 3.5 keep-alive 持续占用内存
- **位置**: `frontend/src/router/index.js:13-14`, `frontend/src/components/Layout.vue`
- **问题**: `ChemSearch` 和 `AdvancedSearch` 均设置 `meta: { keepAlive: true }`，其完整状态（文件数组、Plotly 实例、轮询 interval）在路由切换后仍驻留内存。
- **影响**: 长时间使用后内存占用增大。
- **建议**: 设置 `max` 限制缓存的组件实例数，或在 `onDeactivated` 中清理 Plotly/轮询。

#### 3.6 中英文混杂无国际化框架
- **位置**: 全局前端组件（见 ChemSearch、AdvancedSearch、Login、HistoryRecords 等）
- **问题**: UI 字符串中英文混用，无 i18n 框架，无法切换语言。
- **建议**: 引入 `vue-i18n`，统一管理多语言字符串。

---

### 4. 可观测性

#### 4.1 无请求追踪
- **问题**: 无 request ID 生成/传递，出问题时无法串联日志。
- **建议**: 添加 `X-Request-ID` 中间件。

#### 4.2 无 GPU Worker 监控
- **问题**: GPU 推理耗时、队列深度、处理速率等指标均未暴露。
- **建议**: 添加 Prometheus metrics endpoint，暴露队列深度、作业耗时分布、Worker 状态。

#### 4.3 无错误追踪
- **问题**: 异常仅通过 HTTP 500 返回，未记录到任何错误追踪系统。
- **建议**: 集成 Sentry 或至少记录异常栈到日志文件。

---

### 5. 安全增强

#### 5.1 无 CSP 头
- **问题**: 未设置 Content-Security-Policy，XSS 攻击面大。
- **建议**: 添加严格的 CSP 头。

#### 5.2 无 Token 撤销机制
- **位置**: `backend/core/security.py:19-24`
- **问题**: JWT 无状态，签发后无法撤销。用户修改密码后旧 token 仍有效。
- **建议**: 维护 token 黑名单（Redis）或使用 refresh token 轮换机制。

#### 5.3 无用户角色/权限系统
- **位置**: `backend/models/user.py`
- **问题**: 仅有 User 模型，无角色（admin/regular）区分。
- **建议**: 添加 `role` 字段，admin 可查看所有用户任务、管理 GPU 资源。

#### 5.4 无审计日志
- **问题**: 无用户操作记录（登录、上传、检索、下载），安全事件无法追溯。
- **建议**: 记录关键操作的时间、用户、IP、操作类型和结果。

---

## 问题统计

| 类别 | 急需解决 | 后期升级 |
|------|---------|---------|
| 安全漏洞 | 8 | 4 |
| 并发与线程安全 | 6 | 0 |
| 阻塞与性能 | 3 | 0 |
| 用户数据管理 | 3 | 0 |
| 前端问题 | 7 | 0 |
| 架构与代码质量 | 0 | 7 |
| 后端架构 | 0 | 5 |
| 用户体验 | 0 | 6 |
| 可观测性 | 0 | 3 |
| **合计** | **27** | **25** |
