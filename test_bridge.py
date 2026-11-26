import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from bridge import Bridge, BridgeError, scan


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / 'bridge'
        self.vault = self.base / 'vault'
        self.remote = self.base / 'remote.git'
        self.root.mkdir()
        (self.vault / 'tools').mkdir(parents=True)
        self.note = self.vault / 'tools/note.md'
        self.note.write_text('# Source\n\n[Home](../Home.md)\ninternal.example\n')
        (self.root / 'export.json').write_text(json.dumps({
            'repository': 'test/bridge', 'branch': 'main', 'files': ['tools/note.md']}))
        (self.root / '.bridge-local.json').write_text(json.dumps({
            'vault': str(self.vault), 'replacements': [['internal.example', 'example.com']]}))
        (self.root / '.gitignore').write_text('.bridge-local.json\n')
        (self.root / 'README.md').write_text('# Instructions\n')
        self.run_git(self.root, 'init', '-b', 'main')
        self.run_git(self.base, 'init', '--bare', str(self.remote))
        self.run_git(self.root, 'config', 'user.name', 'Example')
        self.run_git(self.root, 'config', 'user.email', 'example@example.com')
        self.run_git(self.root, 'remote', 'add', 'origin', 'https://github.com/test/bridge.git')
        self.run_git(self.root, 'add', '.gitignore', 'README.md', 'export.json')
        self.run_git(self.root, 'commit', '-m', 'Initial infrastructure')
        self.real_run = subprocess.run
        self.private = True
        self.fail_push = False
        self.runner = patch('bridge.subprocess.run', side_effect=self.intercept)
        self.runner.start()
        self.addCleanup(self.runner.stop)
        self.bridge = Bridge(self.root)

    def intercept(self, args, **kwargs):
        if args[0] == 'gh':
            return subprocess.CompletedProcess(args, 0, 'true\n' if self.private else 'false\n', '')
        if self.fail_push and 'push' in args:
            self.fail_push = False
            return subprocess.CompletedProcess(args, 1, '', 'offline')
        if args[0] == 'git' and ('fetch' in args or 'push' in args):
            args = ['git', '-c', f'url.{self.remote}.insteadOf=https://github.com/test/bridge.git', *args[1:]]
        return self.real_run(args, **kwargs)

    def run_git(self, path, *args):
        return subprocess.run(['git', '-C', str(path), *args], check=True,
                              capture_output=True, text=True).stdout.strip()

    def sync(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.bridge.sync()

    def snapshot(self, root):
        return {str(p.relative_to(root)): p.read_bytes()
                for p in root.rglob('*') if p.is_file() and '.git' not in p.relative_to(root).parts}

    def test_preview_is_read_only_and_sanitized(self):
        before = self.snapshot(self.root), self.snapshot(self.vault)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.bridge.preview()
        self.assertNotIn('internal.example', out.getvalue())
        self.assertIn('example.com', out.getvalue())
        self.assertEqual(before, (self.snapshot(self.root), self.snapshot(self.vault)))

    def test_sync_updates_and_noop_keeps_commit_and_source(self):
        original = self.snapshot(self.vault)
        (self.root / 'unrelated.md').write_text('Not allowed')
        self.sync()
        head = self.run_git(self.root, 'rev-parse', 'HEAD')
        exported = self.snapshot(self.root)
        self.sync()
        self.assertEqual(head, self.run_git(self.root, 'rev-parse', 'HEAD'))
        self.assertEqual(exported, self.snapshot(self.root))
        self.assertEqual(original, self.snapshot(self.vault))
        tracked = self.run_git(self.root, 'ls-tree', '-r', '--name-only', 'origin/main')
        self.assertNotIn('unrelated.md', tracked)
        self.assertNotIn('.bridge-local.json', tracked)
        self.note.write_text('# Updated source\n')
        self.sync()
        self.assertEqual('# Updated source\n', (self.root / 'tools/note.md').read_text())

    def test_failed_push_retries_same_commit(self):
        self.fail_push = True
        with self.assertRaises(BridgeError):
            self.sync()
        pending = self.run_git(self.root, 'rev-parse', 'HEAD')
        self.sync()
        self.assertEqual(pending, self.run_git(self.root, 'rev-parse', 'origin/main'))

    def test_missing_source_does_not_delete_export(self):
        self.sync()
        old = (self.root / 'tools/note.md').read_bytes()
        self.note.unlink()
        with self.assertRaises(BridgeError):
            self.sync()
        self.assertEqual(old, (self.root / 'tools/note.md').read_bytes())

    def test_secrets_block_without_echoing_value(self):
        token = 'ghp_' + 'a' * 36
        self.note.write_text(token)
        with self.assertRaises(BridgeError) as error:
            self.bridge.preview()
        self.assertNotIn(token, str(error.exception))

    def test_source_and_target_symlinks_block(self):
        outside = self.base / 'outside.md'
        outside.write_text('outside')
        self.note.unlink()
        self.note.symlink_to(outside)
        with self.assertRaises(BridgeError):
            self.sync()
        self.note.unlink()
        self.note.write_text('source')
        (self.root / 'tools').symlink_to(self.vault / 'tools', target_is_directory=True)
        with self.assertRaises(BridgeError):
            self.sync()
        self.assertEqual('source', self.note.read_text())

    def test_staged_unrelated_file_blocks(self):
        (self.root / 'unrelated.md').write_text('extra')
        self.run_git(self.root, 'add', 'unrelated.md')
        with self.assertRaises(BridgeError):
            self.sync()

    def test_public_repository_blocks_before_export(self):
        self.private = False
        with self.assertRaises(BridgeError):
            self.sync()
        self.assertFalse((self.root / 'Home.md').exists())

    def test_explicit_public_configuration_allows_public_repository(self):
        config = json.loads((self.root / 'export.json').read_text())
        config['visibility'] = 'public'
        (self.root / 'export.json').write_text(json.dumps(config))
        self.run_git(self.root, 'add', 'export.json')
        self.run_git(self.root, 'commit', '-m', 'Explicitly configure public export')
        self.bridge = Bridge(self.root)
        self.private = False
        self.sync()
        self.assertIn('Repository visibility: public', (self.root / 'Home.md').read_text())
        self.private = True
        with self.assertRaises(BridgeError):
            self.sync()

    def test_remote_ahead_blocks_without_overwriting(self):
        self.sync()
        other = self.base / 'other'
        self.run_git(self.base, 'clone', '-b', 'main', str(self.remote), str(other))
        self.run_git(other, 'config', 'user.name', 'Example')
        self.run_git(other, 'config', 'user.email', 'example@example.com')
        (other / 'README.md').write_text('remote edit\n')
        self.run_git(other, 'add', 'README.md')
        self.run_git(other, 'commit', '-m', 'Remote change')
        self.run_git(other, 'push', 'origin', 'main')
        before = self.snapshot(self.root)
        self.note.write_text('new source\n')
        with self.assertRaises(BridgeError):
            self.sync()
        self.assertEqual(before, self.snapshot(self.root))

    def test_removed_secret_in_pending_history_still_blocks(self):
        self.sync()
        (self.root / 'README.md').write_text('ghp_' + 'b' * 36)
        self.run_git(self.root, 'add', 'README.md')
        self.run_git(self.root, 'commit', '-m', 'Bad pending commit')
        (self.root / 'README.md').write_text('removed\n')
        self.run_git(self.root, 'add', 'README.md')
        self.run_git(self.root, 'commit', '-m', 'Remove value')
        with self.assertRaises(BridgeError):
            self.sync()

    def test_outside_links_do_not_expand_scope_and_examples_stay_examples(self):
        self.note.write_text('[Hidden](../private.md)\n```md\n[Example](../private.md)\n```\n')
        self.sync()
        exported = (self.root / 'tools/note.md').read_text()
        self.assertIn('Hidden (not exported: missing source)', exported)
        self.assertIn('[Example](../private.md)', exported)
        self.assertFalse((self.root / 'private.md').exists())

    def set_files(self, files, **extra):
        config = json.loads((self.root / 'export.json').read_text())
        config.update(files=files, **extra)
        (self.root / 'export.json').write_text(json.dumps(config))
        self.run_git(self.root, 'add', 'export.json')
        self.run_git(self.root, 'commit', '-m', 'Review explicit source scope')
        self.bridge = Bridge(self.root)

    def test_hard_exclusions_cannot_be_overridden(self):
        config = json.loads((self.root / 'export.json').read_text())
        forbidden = ['skills/外部skill/anything.md', 'tools/claude-network/operations.md',
                     '_assets/claude/diagnostic.json', '.claudian/sessions/chat.json',
                     '.obsidian/settings.json', 'todolist/inbox.md', 'tools/linux_commands.md',
                     'projects/claude-workspace/memory.md', 'runbooks/internal.md',
                     'skills/point-cloud-training/references/adapointr-case.md',
                     '3d_learning/__pycache__/cache.py', '_assets/3d_learning/weights.pt']
        for name in forbidden:
            with self.subTest(name=name):
                (self.root / 'export.json').write_text(json.dumps({**config, 'files':[name], 'exclude':[]}))
                with self.assertRaises(BridgeError):
                    Bridge(self.root)

    def test_script_and_attachments_are_exact_bytes(self):
        contents = {
            '3d_learning/day1/demo.py': b'print("example")\r\n',
            '_assets/3d_learning/day1/result.json': b'{"metric": 0.5}\r\n',
            '_assets/3d_learning/day1/result.csv': b'name,value\r\nexample,1\r\n',
            '_assets/3d_learning/day1/figure.png': b'\x89PNG\r\n\x1a\n\x00\xffexample-image',
        }
        for name, data in contents.items():
            path=self.vault / name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(data)
        self.set_files(['tools/note.md', *contents])
        before=self.snapshot(self.vault)
        self.sync()
        for name,data in contents.items():
            self.assertEqual(data, (self.root/name).read_bytes())
        self.assertEqual(before,self.snapshot(self.vault))
        state=json.loads((self.root/'export-state.json').read_text())
        from bridge import digest
        self.assertEqual(digest(contents['_assets/3d_learning/day1/figure.png']),
                         state['files']['_assets/3d_learning/day1/figure.png'])

    def test_unresolved_internal_information_blocks_without_echo(self):
        self.note.write_text('/Users/' + 'somebody/private/path')
        with self.assertRaises(BridgeError) as error:
            self.bridge.preview()
        self.assertNotIn('somebody', str(error.exception))

    def test_utf8_prose_credentials_are_detected_in_bytes(self):
        with self.assertRaises(BridgeError):
            scan('example.md', ('密码' + ' sample123').encode('utf-8'))

    def test_binary_credential_metadata_is_blocked(self):
        name='_assets/3d_learning/day1/figure.png'
        p=self.vault/name; p.parent.mkdir(parents=True)
        p.write_bytes(b'\x89PNG\xff' + ('ghp_' + 'z'*36).encode())
        self.set_files(['tools/note.md',name])
        with self.assertRaises(BridgeError):
            self.sync()
        self.assertFalse((self.root/'Home.md').exists())

    def test_links_cannot_expand_allowlist_or_escape_vault(self):
        self.note.write_text('[Outside](../../outside.md)\n[Excluded](../tools/linux_commands.md)\n')
        self.sync()
        text=(self.root/'tools/note.md').read_text()
        self.assertIn('outside vault',text)
        self.assertFalse((self.root/'tools/linux_commands.md').exists())
        self.assertTrue(self.bridge.report['excluded_references'])

    def test_custom_exclusion_beats_explicit_allowlist(self):
        config=json.loads((self.root/'export.json').read_text())
        config['exclude']=['tools/**']
        (self.root/'export.json').write_text(json.dumps(config))
        with self.assertRaises(BridgeError):
            Bridge(self.root)

    def test_remote_divergence_preserves_both_histories(self):
        self.sync()
        other=self.base/'other'
        self.run_git(self.base,'clone','-b','main',str(self.remote),str(other))
        self.run_git(other,'config','user.name','Example')
        self.run_git(other,'config','user.email','example@example.com')
        (other/'README.md').write_text('remote version')
        self.run_git(other,'add','README.md'); self.run_git(other,'commit','-m','Remote change')
        self.run_git(other,'push','origin','main')
        (self.root/'README.md').write_text('local version')
        self.run_git(self.root,'add','README.md'); self.run_git(self.root,'commit','-m','Local change')
        head=self.run_git(self.root,'rev-parse','HEAD')
        with self.assertRaises(BridgeError):
            self.sync()
        self.assertEqual(head,self.run_git(self.root,'rev-parse','HEAD'))
        self.assertEqual('local version',(self.root/'README.md').read_text())


if __name__ == '__main__':
    unittest.main()
