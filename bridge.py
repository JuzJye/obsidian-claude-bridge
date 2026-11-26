#!/usr/bin/env python3
"""Explicit, one-way Markdown export. Never write to the source vault."""
import argparse
import difflib
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from datetime import datetime
from urllib.parse import unquote, urlsplit
from zoneinfo import ZoneInfo


class BridgeError(Exception):
    pass


INFRA = {'.gitignore', 'README.md', 'bridge.py', 'sync', 'export.json', 'test_bridge.py'}
STATE = 'export-state.json'
GENERATED = {'Home.md', STATE}
SECRET_RULES = [
    ('private key', r'-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----'),
    ('GitHub token', r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b'),
    ('API key', r'\bsk-(?:ant-)?[A-Za-z0-9_-]{20,}\b'),
    ('AWS access key', r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    ('credential URL', r'https?://[^\s/:@]+:[^\s/@]+@'),
    ('credential assignment', r'''(?im)^\s*(?:export\s+)?(?:[A-Z_]*(?:TOKEN|PASSWORD|SECRET|API_KEY))\s*[:=]\s*["']?(?!REPLACE_ME\b|EXAMPLE\b|\$|["']?\s*$)[A-Za-z0-9_+/=-]{12,}'''),
]


def scan(name, data):
    for label, pattern in SECRET_RULES:
        if re.search(pattern, data):
            raise BridgeError(f'{name}: 检测到 {label}；停止，不输出匹配正文。')


def digest(data):
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def json_text(data):
    return json.dumps(data, ensure_ascii=False, indent=2) + '\n'


def unfenced_lines(text):
    fence = None
    for line in text.splitlines(keepends=True):
        match = re.match(r'^\s*(`{3,}|~{3,})', line)
        if match:
            marker = match.group(1)
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
            yield line, False
        else:
            yield line, fence is None


def rewrite_links(text, name, selected):
    def replace(match):
        label, target = match.group(1), match.group(2)
        parts = urlsplit(target)
        if parts.scheme in {'http', 'https', 'mailto'} or target.startswith('#'):
            return match.group(0)
        if parts.scheme or target.startswith('/'):
            return f'{label}（未导出链接）'
        path = PurePosixPath(os.path.normpath(str(PurePosixPath(name).parent / unquote(parts.path))))
        if str(path) == 'Home.md':
            return match.group(0)
        if str(path) not in selected:
            return f'{label}（关联内容未导出）'
        return match.group(0)
    return ''.join(re.sub(r'!?\[([^\]\n]*)\]\(([^\s)]+)\)', replace, line) if active else line
                   for line, active in unfenced_lines(text))


class Bridge:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.config = json.loads((self.root / 'export.json').read_text())
        self.local = json.loads((self.root / '.bridge-local.json').read_text())
        self.vault = Path(self.local['vault']).expanduser().resolve(strict=True)
        if self.root == self.vault or self.vault in self.root.parents or self.root in self.vault.parents:
            raise BridgeError('桥接目录与 vault 必须相互独立。')
        self.files = self.config['files']
        self.visibility = self.config.get('visibility', 'private')
        if self.visibility not in {'private', 'public'}:
            raise BridgeError('visibility must be private or public.')
        if not self.files or len(set(self.files)) != len(self.files):
            raise BridgeError('清单必须非空且不能重复。')
        if self.config['branch'] != 'main' or not re.fullmatch(r'[\w.-]+/[\w.-]+', self.config['repository']):
            raise BridgeError('仓库配置无效。')
        for name in self.files:
            path = PurePosixPath(name)
            if (not isinstance(name, str) or str(path) != name or path.is_absolute()
                    or any(p.startswith('.') for p in path.parts)
                    or path.suffix != '.md' or path.parts[0] in {'skills', '_meta', '_archive'}
                    or name in INFRA | GENERATED):
                raise BridgeError('清单仅接受非隐藏目录下的普通 Markdown 相对路径。')
        self.allowed = INFRA | GENERATED | set(self.files)

    def git(self, *args, check=True):
        result = subprocess.run(['git', '-C', str(self.root), *args], capture_output=True, text=True, timeout=60)
        if check and result.returncode:
            # Git errors can include remote URLs. Do not emit credential-bearing text.
            raise BridgeError(f'Git {args[0]} 失败（退出码 {result.returncode}）。本地内容保留。')
        return result

    def build(self):
        notes = {}
        for name in self.files:
            source = self.vault / name
            if any(p.is_symlink() for p in [source, *source.parents] if p != self.vault):
                raise BridgeError(f'{name}: 不接受符号链接。')
            if not source.is_file() or not source.resolve().is_relative_to(self.vault):
                raise BridgeError(f'{name}: 源文件缺失或越界；停止，不自动删除副本。')
            text = source.read_text(encoding='utf-8')
            scan(name, text)
            for old, new in self.local.get('replacements', []):
                if not old:
                    raise BridgeError('脱敏替换的源字符串不能为空。')
                text = text.replace(old, new)
            text = rewrite_links(text, name, set(self.files))
            scan(name, text)
            notes[name] = text
        hashes = {name: digest(text) for name, text in sorted(notes.items())}
        prior = self.root / STATE
        state = json.loads(prior.read_text()) if prior.exists() else {}
        if set(state.get('files', {})) - set(hashes):
            raise BridgeError('清单缩小；请先审阅旧导出副本的撤下操作。')
        timestamp = state.get('exported_at') if state.get('files') == hashes else None
        timestamp = timestamp or datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(timespec='seconds')
        home = '# Selected notes for Claude\n\n'
        home += f'Exported at: {timestamp} (Asia/Tokyo). This is the export time of these note copies.\n\n'
        home += f'Repository visibility: {self.visibility}.\n\n'
        home += 'Obsidian is the source of truth. These are selected copies with sanitized examples; other vault content is excluded.\n\n'
        home += ''.join(f'- [{name}]({name})\n' for name in self.files)
        home += '\nPlease respond in English, citing the note path, suggested change, and reason. Local Codex reviews feedback before updating the source vault.\n'
        home += '\n[Sync and collaboration guide](README.md)\n'
        notes['Home.md'] = home
        notes[STATE] = json_text({'exported_at': timestamp, 'files': hashes})
        return notes

    def preview(self):
        outputs = self.build()
        changed = False
        for name, text in outputs.items():
            dest = self.root / name
            if any(p.is_symlink() for p in [dest, *dest.parents]):
                raise BridgeError(f'{name}: 目标不能是符号链接。')
            old = dest.read_text() if dest.exists() else ''
            scan(name, old)
            if old != text:
                changed = True
                print(''.join(difflib.unified_diff(old.splitlines(True), text.splitlines(True),
                                                 fromfile=f'current/{name}', tofile=f'export/{name}')), end='')
        if not changed:
            print('无导出变化。')

    def remote_check(self):
        repo = self.config['repository']
        remote = self.git('remote', 'get-url', 'origin').stdout.strip()
        if remote not in {f'https://github.com/{repo}.git', f'https://github.com/{repo}', f'git@github.com:{repo}.git'}:
            raise BridgeError('origin 与配置不一致；停止。')
        result = subprocess.run(['gh', 'api', f'repos/{repo}', '--jq', '.private'],
                                capture_output=True, text=True, timeout=30)
        expected_private = 'true' if self.visibility == 'private' else 'false'
        if result.returncode or result.stdout.strip() != expected_private:
            raise BridgeError(f'GitHub visibility does not match configured {self.visibility}; stopped.')
        self.git('fetch', '--prune', 'origin')
        branch = self.git('symbolic-ref', '--short', 'HEAD').stdout.strip()
        if branch != 'main':
            raise BridgeError('当前分支不是 main；停止。')
        if self.git('show-ref', '--verify', '--quiet', 'refs/remotes/origin/main', check=False).returncode == 0:
            if self.git('merge-base', '--is-ancestor', 'origin/main', 'HEAD', check=False).returncode:
                raise BridgeError('远端领先或分叉；停止，不覆盖、不自动合并。')

    def validate_checkout(self):
        tracked = set(self.git('ls-files', '-z').stdout.split('\0')) - {''}
        if tracked - self.allowed:
            raise BridgeError('存在范围外的已跟踪文件；停止。')
        if self.git('diff', '--cached', '--name-only').stdout.strip():
            raise BridgeError('暂存区非空；请先审阅并提交现有改动。')
        if self.git('diff', '--name-only').stdout.strip():
            raise BridgeError('存在直接修改的副本或实现文件；请先审阅处理。')
        if not self.git('rev-parse', '--verify', 'HEAD', check=False).returncode == 0:
            raise BridgeError('请先审阅并提交桥接工具的初始实现。')
        # Scan every pending commit, including files later removed, before any push.
        remote = self.git('rev-parse', '--verify', 'origin/main', check=False)
        rev_range = 'origin/main..HEAD' if remote.returncode == 0 else 'HEAD'
        commits = self.git('rev-list', rev_range).stdout.splitlines()
        for commit in commits:
            entries = self.git('ls-tree', '-r', commit).stdout.splitlines()
            for entry in entries:
                meta, name = entry.split('\t', 1)
                if meta.split()[0] != '100644' and meta.split()[0] != '100755':
                    raise BridgeError('历史提交包含符号链接或子模块；停止。')
                if name not in self.allowed:
                    raise BridgeError('历史提交包含范围外文件；停止。')
                scan(name, self.git('show', f'{commit}:{name}').stdout)

    def sync(self):
        git_dir = self.root / '.git'
        if not git_dir.is_dir():
            raise BridgeError('此目录尚未初始化独立仓库。')
        with (git_dir / 'bridge.lock').open('w') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise BridgeError('另一同步正在运行。')
            outputs = self.build()
            self.remote_check()
            self.validate_checkout()
            for name, data in outputs.items():
                target = self.root / name
                if any(p.is_symlink() for p in [target, *target.parents]):
                    raise BridgeError(f'{name}: 目标不能是符号链接。')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(data, encoding='utf-8')
            self.git('add', '--', *outputs)
            if self.git('diff', '--cached', '--quiet', check=False).returncode:
                self.git('commit', '-m', 'Sync selected Obsidian notes')
                print('已提交选定笔记。')
            else:
                print('无导出变化，不创建提交。')
            self.git('push', '-u', 'origin', 'HEAD:main')
            print(f"Pushed {self.visibility} repository: https://github.com/{self.config['repository']}")
            print(f"提交：{self.git('rev-parse', 'HEAD').stdout.strip()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preview', 'sync'])
    args = parser.parse_args()
    try:
        bridge = Bridge(Path(__file__).resolve().parent)
        getattr(bridge, args.command)()
    except (BridgeError, OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        # OSError/JSON errors may expose local paths but never print file contents.
        print(f'停止：{exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
