---
name: submit
description: 将指定文件夹下的代码变更提交到 git 仓库
---

# 代码提交 Skill

## 触发方式

`/submit <文件夹名称>`

例如：`/submit backend` 或 `/submit frontend/src/components`

## 执行流程

1. **确认参数**：从用户消息中提取文件夹名称。如果未提供，提示用户并终止。

2. **检查变更**：运行 `git status` 查看仓库状态，运行 `git diff --name-only` 查看已修改文件，确认指定文件夹下有待提交的变更。

3. **仅处理指定文件夹**：只 stage 该文件夹下的文件，使用 `git add <文件夹>/` 或逐个文件添加。

4. **生成 commit message**：根据文件变更内容，用中文撰写 commit message，格式为 `<type>: <简要描述>`，type 包括：
   - `feat`: 新功能
   - `fix`: 修复缺陷
   - `refactor`: 重构
   - `style`: 样式调整
   - `docs`: 文档变更
   - `chore`: 杂项

5. **创建提交**：使用 `git commit` 创建提交。

6. **输出结果**：报告提交成功，包括 commit hash 和变更文件列表。

## 约束

- 不要提交包含密钥的文件（`.env`、`credentials.json` 等）
- 如果指定文件夹下没有变更，提示用户并终止
- 不跳过 git hooks（不使用 `--no-verify`）
- 不 amend 已有提交
- 提交前向用户展示将要提交的文件列表和 commit message，等待确认
