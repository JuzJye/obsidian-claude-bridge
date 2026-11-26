#!/usr/bin/env python3
"""Export reviewed vault files to GitHub; never write to the source vault."""
import argparse
from collections import Counter
from datetime import datetime
import difflib
import fcntl
from fnmatch import fnmatchcase
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from urllib.parse import quote, unquote, urlsplit
from zoneinfo import ZoneInfo


class BridgeError(Exception):
    pass


INFRA = {'.gitignore', 'README.md', 'bridge.py', 'sync', 'export.json', 'test_bridge.py'}
STATE = 'export-state.json'
GENERATED = {'Home.md', STATE}
TEXT_SUFFIXES = {'.md', '.py', '.json', '.csv', '.svg'}
ALLOWED_SUFFIXES = TEXT_SUFFIXES | {'.png', '.jpg', '.jpeg', '.webp'}
# These cannot be overridden by a config allowlist.
HARD_EXCLUDES = (
    'skills/外部skill/**', 'tools/claude-network/**', '_assets/claude/**',
    '_archive/**', '_meta/**', 'todolist/**', 'schedule/**', 'thoughts/**',
    'report/**', 'amendment/**', 'runbooks/**', 'tools/claude-continuity/**',
    'tools/linux_commands.md', 'tools/claude_deployment_mac.md',
    'tools/claude_conversation_backup.md', 'tools/claude_selected_text_clipper.json',
    'skills/remote-work/**', 'skills/mac-connect/**', 'skills/obsidian-vault-manager/**',
    'skills/point-cloud-training/references/adapointr*',
    'PLAN.md', '4090_EXECUTION.md', 'MAC_BACKGROUND_HANDOFF*', 'AGENTS.md',
    '**/__pycache__/**', '*.pyc', '*.pt', '*.pth', '*.onnx', '*.npz', '*.npy',
    '*.wav', '*.mp3', '*.mp4', '*.pdf', '*.zip', '*.7z', '*.log',
)
SECRET_RULES = [
    ('private key', r'-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----'),
    ('GitHub token', r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b'),
    ('API key', r'\bsk-(?:ant-)?[A-Za-z0-9_-]{20,}\b'),
    ('AWS access key', r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    ('credential URL', r'https?://[^\s/:@]+:[^\s/@]+@'),
    ('credential assignment', r'''(?im)^\s*(?:export\s+)?(?:[A-Z_]*(?:TOKEN|PASSWORD|SECRET|API_KEY))\s*[:=]\s*["']?(?!REPLACE_ME\b|EXAMPLE\b|\$|["']?\s*$)[A-Za-z0-9_+/=-]{12,}'''),
    ('password in prose', r'(?:密码|口令)\s*[:：=]?\s*(?!REPLACE_ME\b|EXAMPLE\b)[A-Za-z0-9@_!.-]{4,}'),
]
ENV_RULES = [
    ('internal domain or account', r'\broboscience\.(?:xyz|ai)\b|\b(?:CNF|CNS)[0-9]{5,}\b'),
    ('private network address', r'\b(?:10\.(?:\d{1,3}\.){2}\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b'),
    ('machine-specific user path', r'/Users/[^\s/`"<>]+|/home/(?:nvidia|zzj|CNF[0-9]+|CNS[0-9]+)(?:/|\b)'),
]
LINK = re.compile(r'(!?)\[([^\]\n]*)\]\((<[^>]+>|[^\s)]+)\)')


def payload(data):
    return data.encode('utf-8') if isinstance(data, str) else data


def scan(name, data, environment=False):
    if isinstance(data, str):
        text = data
    else:
        try:
            text = data.decode('utf-8')
        except UnicodeDecodeError:
            text = data.decode('latin-1')
    for label, pattern in SECRET_RULES + (ENV_RULES if environment else []):
        if re.search(pattern, text, flags=re.ASCII):
            raise BridgeError(f'{name}: detected {label}; stopped without echoing matched content.')


def digest(data):
    return hashlib.sha256(payload(data)).hexdigest()


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


def exclusion(name, patterns=()):
    path = PurePosixPath(name)
    if any(p.startswith('.') or p == '__pycache__' for p in path.parts):
        return 'hidden application state or cache'
    if path.parts and path.parts[0] == 'projects' and name != 'projects/claude-workspace/handoff.md':
        return 'private project or unreviewed conversation/memory'
    for pattern in (*HARD_EXCLUDES, *patterns):
        if fnmatchcase(name, pattern):
            return 'excluded by publication policy'
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        return 'unsupported or generated file type'
    return None


def rewrite_links(text, name, selected, report=None, root=None, patterns=()):
    def replace(match):
        target = match.group(3).strip('<>')
        parts = urlsplit(target)
        if parts.scheme in {'http', 'https', 'mailto'} or target.startswith('#'):
            return match.group(0)
        label = match.group(2)
        if parts.scheme or target.startswith('/'):
            reason = 'absolute or local-only link'
            resolved = target
        else:
            resolved = os.path.normpath(str(PurePosixPath(name).parent / unquote(parts.path)))
            if resolved in selected or resolved == 'Home.md':
                return match.group(0)
            if resolved == '..' or resolved.startswith('../'):
                reason = 'outside vault'
            elif root is not None and not (root / resolved).exists():
                reason = 'missing source'
            else:
                reason = exclusion(resolved, patterns) or 'not explicitly selected'
        if report is not None:
            key = 'missing_references' if reason == 'missing source' else 'excluded_references'
            entry = {'source': name, 'target': resolved, 'reason': reason}
            if entry not in report[key]:
                report[key].append(entry)
        return f'{label} (not exported: {reason})'
    # Inline-code examples, like fenced examples, are not navigation links.
    return ''.join(''.join(piece if piece.startswith('`') else LINK.sub(replace, piece)
                           for piece in re.split(r'(`+[^`]*`+)', line)) if active else line
                   for line, active in unfenced_lines(text))


class Bridge:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.config = json.loads((self.root / 'export.json').read_text())
        self.local = json.loads((self.root / '.bridge-local.json').read_text())
        self.vault = Path(self.local['vault']).expanduser().resolve(strict=True)
        if self.root == self.vault or self.vault in self.root.parents or self.root in self.vault.parents:
            raise BridgeError('The checkout and source vault must be separate.')
        self.files = self.config['files']
        self.visibility = self.config.get('visibility', 'private')
        self.excludes = self.config.get('exclude', [])
        if self.visibility not in {'private', 'public'}:
            raise BridgeError('visibility must be private or public.')
        if not isinstance(self.files, list) or not self.files or any(not isinstance(n, str) for n in self.files):
            raise BridgeError('files must be a non-empty explicit list of paths.')
        if len(set(self.files)) != len(self.files):
            raise BridgeError('Duplicate allowlist entries.')
        if not isinstance(self.excludes, list) or any(not isinstance(p, str) for p in self.excludes):
            raise BridgeError('exclude must be a list of patterns.')
        if self.config['branch'] != 'main' or not re.fullmatch(r'[\w.-]+/[\w.-]+', self.config['repository']):
            raise BridgeError('Invalid repository configuration.')
        for name in self.files:
            path = PurePosixPath(name)
            if str(path) != name or path.is_absolute() or '..' in path.parts or name in INFRA | GENERATED:
                raise BridgeError(f'{name}: invalid source path.')
            reason = exclusion(name, self.excludes)
            if reason:
                raise BridgeError(f'{name}: {reason}; an allowlist cannot override exclusions.')
        self.allowed = INFRA | GENERATED | set(self.files)
        self.report = {}

    def git(self, *args, check=True, binary=False, input=None):
        result = subprocess.run(['git', '-C', str(self.root), *args], capture_output=True,
                                text=not binary, input=input, timeout=60)
        if check and result.returncode:
            raise BridgeError(f'Git {args[0]} failed (exit {result.returncode}); local content preserved.')
        return result

    def build(self):
        notes = {}
        self.report = {'included': sorted(self.files), 'excluded_references': [], 'missing_references': []}
        for name in self.files:
            source = self.vault / name
            if any(p.is_symlink() for p in [source, *source.parents] if p != self.vault):
                raise BridgeError(f'{name}: symlinks are not allowed.')
            if not source.is_file() or not source.resolve().is_relative_to(self.vault):
                raise BridgeError(f'{name}: missing source or path escape; no deletion performed.')
            data = source.read_bytes()
            limit = 1024 * 1024 if source.suffix.lower() in TEXT_SUFFIXES else 10 * 1024 * 1024
            if len(data) > limit:
                raise BridgeError(f'{name}: exceeds the reviewed small-file limit.')
            if source.suffix.lower() == '.md':
                text = data.decode('utf-8')
                scan(name, text)
                for old, new in self.local.get('replacements', []):
                    if not old:
                        raise BridgeError('Empty redaction source string.')
                    text = text.replace(old, new)
                for pattern in self.local.get('markdown_omit_lines', {}).get(name, []):
                    text = re.sub(pattern, '', text, flags=re.MULTILINE)
                text = rewrite_links(text, name, set(self.files), self.report, self.vault, self.excludes)
                scan(name, text, environment=True)
                notes[name] = text
            else:
                # Scripts and evidence files are copied byte-for-byte, never silently edited.
                scan(name, data, environment=True)
                notes[name] = data
        hashes = {name: digest(data) for name, data in sorted(notes.items())}
        prior = self.root / STATE
        state = json.loads(prior.read_text()) if prior.exists() else {}
        if set(state.get('files', {})) - set(hashes):
            raise BridgeError('Allowlist shrank; explicitly review withdrawal of previously exported files.')
        timestamp = state.get('exported_at') if state.get('files') == hashes else None
        timestamp = timestamp or datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(timespec='seconds')
        home = '# Selected knowledge for Claude\n\n'
        home += f'Exported at: {timestamp} (Asia/Tokyo).\n\nRepository visibility: {self.visibility}.\n\n'
        home += 'Obsidian is the source of truth. This is an explicit, sanitized selection, not a vault backup. New files are not automatically published.\n\n'
        base = f"https://raw.githubusercontent.com/{self.config['repository']}/main/"
        home += 'Read this index, resolve relative links against its raw URL, and fetch the relevant linked files. Source-note language is preserved; collaboration requests use English.\n\n'
        groups = [('Work handoff', 'projects/'), ('3D learning', '3d_learning/'),
                  ('Paper notes', 'papers/'), ('Tool notes', 'tools/'), ('Reusable methods', 'skills/')]
        for title, prefix in groups:
            names = [n for n in sorted(self.files) if n.startswith(prefix) and n.endswith('.md')]
            if names:
                home += f'## {title}\n\n'
                for name in names:
                    encoded = quote(name, safe='/._-~')
                    home += f'- [{name}]({encoded}) · [Raw]({base}{encoded})\n'
                home += '\n'
        types = Counter(Path(n).suffix for n in self.files)
        home += '## Supporting files\n\n'
        home += ', '.join(f'{count} {ext}' for ext, count in sorted(types.items())) + '.\n'
        home += 'Exercises and selected figures/results retain their source paths and are linked from the notes. Excluded or unavailable references are labeled explicitly.\n\n'
        home += 'Private operations, original conversations/memory, workplace documents, the inbox, external skill packages, caches, models, audio, and original paper PDFs are not published.\n\n'
        home += '[Sync and collaboration guide](README.md)\n'
        notes['Home.md'] = home
        notes[STATE] = json_text({'exported_at': timestamp, 'files': hashes})
        return notes

    def preview(self, summary=False):
        outputs = self.build()
        print(f'Included: {len(self.files)} explicitly selected source files.')
        if not summary:
            for name in self.report['included']:
                print(f'  INCLUDE {name}')
        skipped = Counter()
        for path in self.vault.rglob('*'):
            if path.is_file():
                name = str(path.relative_to(self.vault))
                if name not in self.files:
                    skipped[exclusion(name, self.excludes) or 'not explicitly selected'] += 1
        print('Excluded/unselected source inventory:', dict(sorted(skipped.items())))
        for key in ['excluded_references', 'missing_references']:
            print(f'{key}: {len(self.report[key])}')
            for entry in self.report[key]:
                print(f"  {entry['source']} -> {entry['target']} ({entry['reason']})")
        changed = []
        for name, data in outputs.items():
            dest = self.root / name
            if any(p.is_symlink() for p in [dest, *dest.parents]):
                raise BridgeError(f'{name}: destination symlink.')
            old = dest.read_bytes() if dest.exists() else b''
            scan(name, old)
            if old != payload(data):
                changed.append(name)
                if summary:
                    continue
                if Path(name).suffix in TEXT_SUFFIXES:
                    print(''.join(difflib.unified_diff(old.decode('utf-8').splitlines(True),
                                payload(data).decode('utf-8').splitlines(True),
                                fromfile=f'current/{name}', tofile=f'export/{name}')), end='')
                else:
                    print(f'  BINARY {name}: {len(payload(data))} bytes, SHA256 {digest(data)}')
        print(f'Changed exports: {len(changed)}' if changed else 'No export changes.')

    def remote_check(self):
        repo = self.config['repository']
        remote = self.git('remote', 'get-url', 'origin').stdout.strip()
        if remote not in {f'https://github.com/{repo}.git', f'https://github.com/{repo}', f'git@github.com:{repo}.git'}:
            raise BridgeError('origin does not match the configured repository.')
        result = subprocess.run(['gh', 'api', f'repos/{repo}', '--jq', '.private'],
                                capture_output=True, text=True, timeout=30)
        expected_private = 'true' if self.visibility == 'private' else 'false'
        if result.returncode or result.stdout.strip() != expected_private:
            raise BridgeError(f'GitHub visibility does not match configured {self.visibility}; stopped.')
        self.git('fetch', '--prune', 'origin')
        if self.git('symbolic-ref', '--short', 'HEAD').stdout.strip() != 'main':
            raise BridgeError('Current branch is not main.')
        if self.git('show-ref', '--verify', '--quiet', 'refs/remotes/origin/main', check=False).returncode == 0:
            if self.git('merge-base', '--is-ancestor', 'origin/main', 'HEAD', check=False).returncode:
                raise BridgeError('Remote is ahead or divergent; no automatic merge or overwrite.')

    def validate_history(self):
        remote = self.git('rev-parse', '--verify', 'origin/main', check=False)
        rev_range = 'origin/main..HEAD' if remote.returncode == 0 else 'HEAD'
        commits = self.git('rev-list', rev_range).stdout.splitlines()
        blobs = {}
        for commit in commits:
            for entry in self.git('ls-tree', '-rz', commit, binary=True).stdout.split(b'\0'):
                if not entry:
                    continue
                meta, rawname = entry.split(b'\t', 1)
                mode, kind, oid = meta.decode().split()
                name = rawname.decode('utf-8')
                if mode not in {'100644', '100755'} or kind != 'blob' or name not in self.allowed:
                    raise BridgeError('Pending history contains out-of-scope paths, symlinks, or submodules.')
                if name not in INFRA | GENERATED and exclusion(name, self.excludes):
                    raise BridgeError('Pending history contains excluded source content.')
                blobs.setdefault(oid, name)
        if blobs:
            data = self.git('cat-file', '--batch', binary=True,
                            input=('\n'.join(blobs)+'\n').encode()).stdout
            offset = 0
            for oid, name in blobs.items():
                end = data.index(b'\n', offset)
                header = data[offset:end].decode().split()
                if header[0] != oid or header[1] != 'blob':
                    raise BridgeError('Unexpected Git object response.')
                size = int(header[2]); start = end + 1
                scan(name, data[start:start+size], environment=name not in INFRA)
                offset = start + size + 1

    def validate_checkout(self):
        tracked = set(self.git('ls-files', '-z').stdout.split('\0')) - {''}
        if tracked - self.allowed:
            raise BridgeError('Tracked files exceed the publication allowlist.')
        if self.git('diff', '--cached', '--name-only').stdout.strip():
            raise BridgeError('Staging area is not empty; review existing changes first.')
        if self.git('diff', '--name-only').stdout.strip():
            raise BridgeError('Direct checkout edits must be reviewed before sync.')
        if self.git('rev-parse', '--verify', 'HEAD', check=False).returncode:
            raise BridgeError('Review and commit initial infrastructure first.')
        self.validate_history()

    def sync(self):
        git_dir = self.root / '.git'
        if not git_dir.is_dir():
            raise BridgeError('Initialize a separate export checkout first.')
        with (git_dir / 'bridge.lock').open('w') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise BridgeError('Another sync is running.')
            outputs = self.build()
            self.remote_check()
            self.validate_checkout()
            for name in outputs:
                target = self.root / name
                if any(p.is_symlink() for p in [target, *target.parents]):
                    raise BridgeError(f'{name}: destination symlink.')
            for name, data in outputs.items():
                target = self.root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload(data))
            self.git('add', '--', *outputs)
            for name, data in outputs.items():
                if self.git('show', f':{name}', binary=True).stdout != payload(data):
                    raise BridgeError('Staged export changed concurrently; review staging before retry.')
            if self.git('diff', '--cached', '--quiet', check=False).returncode:
                self.git('commit', '-m', 'Sync selected Obsidian knowledge')
                print('Committed selected source exports.')
            else:
                print('No export changes; no new commit.')
            self.validate_history()
            self.git('push', '-u', 'origin', 'HEAD:main')
            print(f"Pushed {self.visibility} repository: https://github.com/{self.config['repository']}")
            print(f"Commit: {self.git('rev-parse', 'HEAD').stdout.strip()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preview', 'sync'])
    parser.add_argument('--summary', action='store_true', help='Hide file listing and full diffs in preview')
    args = parser.parse_args()
    try:
        bridge = Bridge(Path(__file__).resolve().parent)
        if args.command == 'preview':
            bridge.preview(summary=args.summary)
        else:
            bridge.sync()
    except (BridgeError, OSError, ValueError, KeyError, UnicodeError, subprocess.TimeoutExpired) as exc:
        print(f'Stopped: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
