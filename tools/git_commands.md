---
type: tool
status: active
created: 2026-09-06
updated: 2026-11-26
tags: [git, cheatsheet]
---

# Git 最常用指令

适用：已安装 Git 的 Linux、macOS、Windows 终端；`switch` / `restore` 需要 Git 2.23+。示例中的仓库地址、文件名和分支名请替换成自己的。

## 第一次使用



配置ssh

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# 只在没有这对密钥时生成，已有则复用
if [ ! -e ~/.ssh/id_ed25519_example ] &&
   [ ! -e ~/.ssh/id_ed25519_example.pub ]; then
    ssh-keygen -t ed25519 \
        -f ~/.ssh/id_ed25519_example \
        -C "user@example.com"
fi

cat ~/.ssh/id_ed25519_example.pub
```

复制输出的整行公用钥匙，添加到gitlab帐号ssh keys，配置别名

```bash
cat >> ~/.ssh/config <<'EOF'

Host gitlab-example
    HostName gitlab.example.com
    User git
    IdentityFile ~/.ssh/id_ed25519_example
    IdentitiesOnly yes
    IdentityAgent none
EOF

chmod 600 ~/.ssh/config
chmod 600 ~/.ssh/id_ed25519_example
```

```bash
# 验证ssh 在服务器的 `/home/example/.ssh/config` 
ssh -T git@gitlab-example
# 应显示：Welcome to GitLab, @EXAMPLE_USER!

# 克隆仓库
mkdir -p ~/calib
cd ~/calib
# 之后拉取都要用gitlab-example的别名
git clone git@gitlab-example:EXAMPLE_GROUP/example-calibration.git

# 验证身份
cd ~/calib/example-calibration
git config --local user.name "示例用户"
git config --local user.email "user@example.com"

# 删除文件夹
rmdir 空文件夹
rm -rf 非空文件夹名
```




==重要==
```bash
git config --local user.name 示例用户
git config --local user.email user@example.com
# 如果ssh认证设置了别人的账户
git config --global user.name 示例用户
git config --global user.email user@example.com
git clone <仓库地址>              # 下载仓库
# 下载子模块，需要输入username为工号，以及个人token
git submodule update --init --recursive
```

```bash
# 如果不是默认分支
git clone -b feat/example git@gitlab.example.com:example-group/example-project.git

# 如果拉错分支了
git fetch origin
git switch feat/example
git pull
git status
cd <仓库目录>
```

## 日常只记这些

| 指令 | 用途 |
| --- | --- |
| `git status` | 查看当前分支、哪些文件有修改 |
| `git diff` | 查看尚未暂存的修改 |
| `git diff --staged` | 查看已暂存、准备提交的修改 |
| `git add 文件名` | 把指定文件加入暂存区 |
| `git add .` | 暂存当前目录及子目录的修改、新增和删除；先确认没有混入无关文件 |
| `git commit -m "修改说明"` | 将暂存内容提交到本地 |
| `git pull --ff-only` | 拉取并快进更新当前分支；分支有分叉时会停止 |
| `git push` | 将本地提交推送到远端 |
| `git log --oneline -10` | 查看最近 10 条提交 |

git rebell 啥东西？？？

常用顺序：**拉取 → 修改 → 查看差异 → 暂存 → 提交 → 推送**。`pull` / `push` 默认要求当前分支已关联远端分支；拉取前先提交或暂存自己的工作。

## 分支与临时收起修改

| 指令                            | 用途                      |
| ----------------------------- | ----------------------- |
| `git branch`                  | 查看本地分支，`*` 表示当前分支       |
| `git switch -c 分支名`           | 新建并切换分支                 |
| `git switch 分支名`              | 切换已有分支                  |
| `git push -u origin 分支名`      | 首次推送新分支，并关联远端分支         |
| `git stash push -u -m "临时保存"` | 临时收起修改，包含未跟踪文件，不包含被忽略文件 |
| `git stash pop`               | 恢复最近一次收起的修改；可能出现冲突      |

### 将最新 `dev` 同步到自己的开发分支

#### 步骤一：同步最新dev

每天开始开发前同步一次，以及准备 MR 前再同步一次即可，不必每次小改动都同步。

```bash
# 1. 前置操作
cd /home/example/projects/example-project
git status
git fetch origin
# 1. 切到本地dev，更新到远端dev
git switch dev
git pull --ff-only origin dev

# 2. 切回个人特性分支，把最新dev合并进来
git switch example-user/feature
git merge dev

# 👉 若出现子模块M标记（子模块hash/uri不一致）
git submodule sync
git submodule update --init

# 冲突处理：
# 解决冲突文件后 git add <file>，最后 git commit 完成merge
# 想放弃本次合并：git merge --abort
```

#### 情况二：硬重置，适合大量冲突，直接对齐origin/dev

**Warning:** `git reset --hard origin/dev` discards uncommitted changes to tracked files and moves the current branch to `origin/dev`. Create a backup branch with `git branch <backup-branch>` to preserve the current commits, and save uncommitted work separately before proceeding. A later force push will also move the remote branch.

```bash
# 1. 切回个人特性分支
git switch example-user/feature

# 2. 硬重置本地分支，完全对齐远端dev
git reset --hard origin/dev

# 3. 消除父仓库层面子模块指针脏标记（仅父仓库标记，不解决子模块远端commit丢失底层问题）
git restore src/submodule/example

# 4. 校验
git status

```

推送指令：
```bash
--force-with-lease
```


#### 情况三：每次改代码前都跟main/dev分支进行同步

```bash
cd /home/example/projects/example-project
git switch feat/example
git status --short
git fetch --prune origin
# git stash push -u
# git stash pop
git merge dev
```



## 撤销

```bash
git restore --staged 文件名      # 取消暂存，保留文件修改
git restore 文件名               # 丢弃该文件未暂存的修改，恢复为暂存区内容
```

**`git restore 文件名` 会丢弃未暂存的改动，执行前先看 `git diff`。**

核对命令：`git --version` 查看版本，`git <命令> -h` 查看帮助。

验证记录（2026-09-10）：使用 Git 2.50.1 在临时本地仓库中跑通了“情况一”和“情况二”的无冲突核心流程；实际项目的远端权限、分支保护规则和冲突处理仍待确认。

[返回首页](../Home.md)
