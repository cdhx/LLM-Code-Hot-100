# 🗳️ 投票系统部署指南

本指南帮助你设置投票后端（Supabase）和部署网站（GitHub Pages）。

## 📋 架构

```
用户 → GitHub Pages (静态网站) → Supabase (数据库)
```

- **前端**：纯静态 HTML/JS，托管在 GitHub Pages
- **后端**：Supabase 提供数据库 + API，无需写服务器代码

---

## 1️⃣ 设置 Supabase（5 分钟）

### Step 1: 创建账号和项目

1. 访问 [supabase.com](https://supabase.com)
2. 用 GitHub 登录
3. 点击 "New Project"
4. 填写项目信息：
   - **Name**: `llm-hot-100`
   - **Database Password**: 设置一个密码（记下来）
   - **Region**: 选择离你近的区域
5. 点击 "Create new project"，等待 1-2 分钟

### Step 2: 创建数据库表

1. 在项目 Dashboard 左侧点击 "SQL Editor"
2. 点击 "New query"
3. 粘贴以下 SQL 并运行：

```sql
-- 创建投票表
create table votes (
  id bigint generated always as identity primary key,
  user_id text not null,
  topic_id text not null,
  created_at timestamp with time zone default now(),
  
  -- 防止重复投票
  unique(user_id, topic_id)
);

-- 创建索引加速查询
create index idx_votes_topic on votes(topic_id);
create index idx_votes_user on votes(user_id);

-- 启用 RLS (Row Level Security)
alter table votes enable row level security;

-- 允许所有人读取
create policy "Allow public read" on votes
  for select using (true);

-- 允许所有人插入（匿名投票）
create policy "Allow public insert" on votes
  for insert with check (true);

-- 允许用户删除自己的投票
create policy "Allow users to delete own votes" on votes
  for delete using (true);
```

4. 点击 "Run" 执行

### Step 3: 获取 API 密钥

1. 左侧点击 "Settings" → "API"
2. 复制以下两个值：
   - **Project URL**: `https://xxxxx.supabase.co`
   - **anon public key**: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`

### Step 4: 配置前端

编辑 `vote.html`，找到这两行（约第 150 行）：

```javascript
const SUPABASE_URL = 'YOUR_SUPABASE_URL';
const SUPABASE_ANON_KEY = 'YOUR_SUPABASE_ANON_KEY';
```

替换为你的值：

```javascript
const SUPABASE_URL = 'https://xxxxx.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...';
```

---

## 2️⃣ 部署到 GitHub Pages（3 分钟）

### Step 1: 推送代码到 GitHub

```bash
git add .
git commit -m "Add voting system"
git push origin main
```

### Step 2: 启用 GitHub Pages

1. 进入仓库 Settings → Pages
2. Source 选择 "Deploy from a branch"
3. Branch 选择 `main`，文件夹选择 `/ (root)`
4. 点击 Save

### Step 3: 访问网站

等待 1-2 分钟后，访问：
```
https://cdhx.github.io/LLM-Code-Hot-100/
```

---

## 3️⃣ 可选：配置自定义域名

如果你有自己的域名（如 `llm-hot-100.com`）：

1. 在仓库根目录创建 `CNAME` 文件，内容为你的域名
2. 在域名 DNS 设置中添加 CNAME 记录指向 `cdhx.github.io`
3. 在 GitHub Pages 设置中填入自定义域名

---

## 🔧 故障排除

### 投票不生效？

1. 打开浏览器控制台（F12）查看错误
2. 确认 Supabase URL 和 Key 正确
3. 确认数据库表已创建

### 跨域问题？

Supabase 默认允许所有来源，如果有问题：
1. Supabase Dashboard → Settings → API
2. 检查 "Additional allowed origins" 设置

### 本地测试

直接打开 `index.html` 或用本地服务器：
```bash
# Python 3
python -m http.server 8000

# Node.js
npx serve
```

---

## 📊 查看投票数据

在 Supabase Dashboard → Table Editor → votes 可以看到所有投票记录。

也可以用 SQL 查询热门题目：
```sql
select topic_id, count(*) as votes
from votes
group by topic_id
order by votes desc
limit 20;
```

---

## 🔒 安全说明

- `anon key` 是公开的，可以放在前端代码中
- RLS 策略保护数据库，只允许指定操作
- 用户 ID 是浏览器生成的随机 ID，不需要登录

---

## 💰 费用说明

Supabase 免费版包含：
- 500 MB 数据库存储
- 2 GB 带宽/月
- 无限 API 请求

对于投票应用，这些额度**完全够用**，不需要付费。
