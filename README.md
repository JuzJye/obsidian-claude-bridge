# Obsidian 与 Claude 的私有资料桥

阅读笔记请从 [Home](Home.md) 开始。这里只保存明确选定的导出副本，Obsidian 是主本。

## 日常使用

在本机告诉 Codex「同步给 Claude」，或在此目录运行：

```sh
./sync preview
./sync sync
```

`preview` 仅显示经过脱敏的预期差异，不修改文件、不联网。`sync` 先验证私有属性与远端状态，再导出、提交和推送；没有变化时不创建提交。首次配置及实现文件的改动需单独审阅并提交；日常命令只暂存清单内的导出文件与导出状态。

## Claude 网页端

1. 在 Claude 设置中连接 GitHub，只授权此仓库。
2. 在聊天的「Add from GitHub」，或 Project knowledge 的 GitHub 入口中，选择 `Home.md` 和需要的 `tools/` 文件。
3. 推送后在 Project knowledge 点击 Sync；普通聊天重新添加更新后的文件。请先核对 Home 中的导出时间。
4. 请 Claude 输出建议、涉及的笔记路径与理由；将有用的回复交给本地 Codex，由 Codex核对后更新 Obsidian，再同步。

这条流程中的 GitHub 集成用于读取文件，回复不会自动写回本仓库。Private 限制公众访问，获授权的 GitHub 与 Claude 仍会处理所选资料。

官方参考：[GitHub 集成与私有仓库授权](https://support.claude.com/en/articles/10167454-use-the-github-integration)。

## 安装与扩展

依赖 macOS/Linux、Python 3.9+、Git、已登录的 GitHub CLI。仅在一台桥接电脑运行，仓库放在 Obsidian vault 外。

首次运行前创建被忽略的 `.bridge-local.json`，填写本机 vault 路径和仅作用于副本的字面量替换：

```json
{
  "vault": "/absolute/path/to/vault",
  "replacements": [["internal.example", "gitlab.example.com"]]
}
```

认证复用 `gh auth login` 与系统凭证存储；不要把 token 写入配置或笔记。仓库 remote 必须是配置对应的 GitHub HTTPS 或 SSH 地址。

扩大范围时先修改 `export.json` 的明确文件清单，检查 `.bridge-local.json` 的脱敏规则，运行 preview；审阅并提交配置后再 sync。只支持显式 Markdown 文件，不自动追踪链接、目录或附件。文件不能是符号链接，也不能位于隐藏目录、skills 或部署配置目录。超出范围的实际链接会改为未导出提示；代码块里的语法示例保持为示例。`Home.md` 是本仓库生成的阅读入口，不是原库首页副本。

同步只读原库。源文件缺失或清单缩小时停止，不自动删除远端旧副本；如需撤下资料，手动审阅删除并重新生成导出状态。移除文件不清除 Git 历史。

## 验证、维护与恢复

```sh
python3 -m unittest -v test_bridge.py
./sync preview
```

推送失败保留本地提交，网络恢复后重跑 sync。远端领先或分叉时停止，不拉取覆盖、不 force push；先人工检查两端变化。导出副本被直接修改时停止，先决定是否将改动整理回原库。凭证扫描是常见格式检测，不能代替上传范围与正文审阅；扫描报错只输出文件名和类型。

更新脚本后运行上述测试，审阅 Git 差异并单独提交实现文件。回滚脚本使用 Git revert；不从本仓库反向覆盖 Obsidian。卸载只需停止使用命令并移走库外 checkout，Obsidian Sync 不受影响。此工具没有定时任务或 Obsidian 插件。
