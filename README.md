# web 项目代码开源与复现说明

本仓库仅包含**源码与必要配置**，不包含大型模型文件、数据库文件与运行时生成数据。

## 1. 仓库内容说明

- 已保留：前端源码、后端源码、依赖配置（`frontend/package.json`、`backend/requirements.txt`）
- 已忽略：模型权重、数据库、日志、缓存、临时目录、打包产物（见 `.gitignore`）

## 2. 环境要求

- Python：建议 3.10+
- Node.js：建议 18+
- npm：建议 9+

## 3. 后端启动（FastAPI）

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

> 如果你的环境是 Conda，也可以改用 Conda 环境安装 `requirements.txt`。

## 4. 前端启动（Vite + Vue）

```bash
cd frontend
npm install
npm run dev
```

默认前端会在 `6006` 端口启动（以 `package.json` 的脚本为准）。

## 5. 模型与数据文件准备（关键）

由于仓库不上传大文件，请在本地手动准备并放入对应位置：

- 模型权重文件（如 `*.pt`、`*.pth`、`*.onnx`）
- 数据库文件（如 `users.db`，如需历史数据）
- 运行时数据目录（如 `backend/user_data/`）

请将这些文件通过网盘/对象存储/内部文件服务器分发，并在本地按项目约定路径放置。

## 6. 推荐开源协作习惯

- 新增依赖时同步更新：
  - Python：`backend/requirements.txt`
  - Node：`frontend/package.json`
- 提交前先检查是否有大文件误加入：

```bash
git status
git add .
git status
```

如果发现不该提交的文件，先 `git restore --staged <文件>` 再检查 `.gitignore`。

## 7. 目录建议

```text
web/
├── backend/
│   ├── requirements.txt
│   └── ...
├── frontend/
│   ├── package.json
│   └── ...
├── .gitignore
└── README.md
```
