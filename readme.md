因为您的后端是通过 AutoDL 的网关（Nginx 反向代理）进行转发的（把 HTTPS 8443 转到了内网 HTTP 6006）。

通常情况下，直接调用 get_remote_address 拿到的可能是代理网关的内网 IP（比如 10.0.x.x），而不是公网上那个用户的真实家庭宽带 IP。如果真的是这样，所有的用户都会被误认为是同一个 IP 发起的请求，导致所有人一起被限流。

---

## 用户系统改造说明（2026-04）

后端已支持多用户与 JWT 鉴权：
1. `POST /register`：注册用户并初始化用户目录（`/root/web/backend/user_data/<username>`）。
2. `POST /login`：登录并获取 `access_token`。
3. 所有业务接口通过 `Authorization: Bearer <token>` 识别当前用户，不再使用前端传 `user_id`。
4. 用户隔离后，任务与历史数据写入各自目录，互不可见。

前端已新增登录页与路由守卫：
- 未登录会跳转 `/login`。
- 登录后自动注入 Bearer Token。

部署注意：
- 请在环境变量设置 `JWT_SECRET_KEY`（强烈建议生产环境配置）。
- 新增依赖：`SQLAlchemy`、`PyJWT`、`passlib[bcrypt]`、`bcrypt`。
