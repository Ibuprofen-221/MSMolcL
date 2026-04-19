# web 项目代码开源与复现说明

本仓库仅包含**源码与必要配置**，不包含大型模型文件、数据库文件与运行时生成数据。
注意，为了运行后端，需要在配置cuda的GPU服务器上运行后端代码

## 1. 内容说明

- 前端源码: frontend
- 后端源码：backend
- 依赖配置: frontend/package.json、backend/requirements.txt

## 2. 环境要求

- Python：3.10
- Node.js：18.08.2
- npm：9.8.1
- 建议使用vscode IDE运行代码，这样能够直接映射前后端端口到本地

## 3. 创建并配置虚拟环境
conda create -n env_name python==3.10
conda activate env_name
pip install -r requirements.txt

## 4. 后端启动（FastAPI）

```bash
conda activate env_name
python main.py 
```
后端默认在6008端口启动

## 5. 前端启动（Vite + Vue）

```bash
cd frontend
npm install
npm run dev
```

默认前端会在 `6006` 端口启动
如果使用vscode的转发端口功能，便能直接在浏览器中访问本地6008端口访问前端：http:127.0.0.1:6008
## 6. 模型与数据文件准备
本代码文件夹不提供较大的模型 / 数据库文件
需要在models_and_dbs.tar.gz中额外进行下载。注意：其中pubchem的数据库文件过大(20G)，请准备好足够的磁盘空间。
为了保持源代码的兼容性，建议您复制其他任意一个数据库文件(parquet)，并改名为Pubchem.parquet

并将上述文件统一放在以下位置：

在当前文件夹的同级文件夹中创建autodl-tmp文件夹，并将全部文件放在该文件夹下
路径：
--autodl/database
          ---pubchem
          ---ymdb
          ---...
        /...pth
        /...pth
--web/
  ├── backend/
  │   ├── requirements.txt
  │   └── ...
  ├── frontend/
  │   ├── package.json
  │   └── ...
  ├── .gitignore
  └── README.md