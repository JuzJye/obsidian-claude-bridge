# Obsidian to Claude: selected public notes

Start with [Home](Home.md). Obsidian remains the source of truth. Only the two explicitly selected notes and the bridge implementation are published here. Internal examples are sanitized in the exported copies; the original vault is separate.

The repository is public at the owner's request. Anyone can read its published files and Git history. Other notes, application settings, credentials, and local redaction configuration are excluded.

## Daily use

Ask local Codex to sync the selected notes, or run these commands in this checkout:

```sh
./sync preview
./sync sync
```

`preview` shows sanitized differences without writing files or using the network. `sync` checks that remote visibility matches the explicit `visibility` setting in `export.json`, then exports, commits, and pushes. An unchanged export creates no new commit. Implementation and configuration edits must be reviewed and committed separately; routine sync stages only selected exports and generated state.

## Reading from Claude

Send requests to Claude in English. Ask it to report an explicit failure rather than infer unread file contents.

Public raw-text links, which can be supplied directly to Claude:

- [Home](https://raw.githubusercontent.com/JuzJye/obsidian-claude-bridge/main/Home.md)
- [Git commands](https://raw.githubusercontent.com/JuzJye/obsidian-claude-bridge/main/tools/git_commands.md)
- [Markdown commands](https://raw.githubusercontent.com/JuzJye/obsidian-claude-bridge/main/tools/markdown_commands.md)

Alternatively, use **Add from GitHub**, choose `JuzJye/obsidian-claude-bridge`, and ask Claude to read the relevant files on `main`. The current chat UI adds a repository URL to the prompt; selecting a repository alone does not prove the chat can retrieve its contents. Verify the export timestamp and titles in an actual response. For Project knowledge, refresh its GitHub sync after pushes when that feature is available.

Review useful feedback with local Codex, update the original Obsidian note, then sync again. Chat replies are not automatically written back to this repository. Public access removes the repository's authentication barrier; actual retrieval still depends on the tools available in the Claude chat.

Official reference: [GitHub integration](https://support.claude.com/en/articles/10167454-use-the-github-integration).

## Setup and scope

Requires macOS/Linux, Python 3.9+, Git, and a signed-in GitHub CLI. Run the bridge from one designated computer, outside the Obsidian vault.

Create the ignored `.bridge-local.json` on that computer:

```json
{
  "vault": "/absolute/path/to/vault",
  "replacements": [["internal.example", "gitlab.example.com"]]
}
```

Authentication uses `gh auth login` and the system credential store. Never put a token in notes or configuration. The Git remote must match the configured GitHub repository. A missing `visibility` setting defaults to `private` for compatibility; this checkout explicitly uses `public`.

To add notes, edit the explicit Markdown file list in `export.json`, review the local redaction rules, run preview, and review and commit configuration changes before sync. Links, folders, and attachments are not automatically included. Symlinks, hidden paths, skills, and policy/archive directories are excluded. Out-of-scope prose links become exclusion notices; fenced syntax examples remain examples. Generated `Home.md` is an export entry page, not a copy of the vault homepage.

The bridge only reads the vault. Missing source files or a smaller allowlist stop the sync instead of removing old copies. Withdrawing published content requires an explicit review; removing the current copy does not erase Git history.

## Verification and recovery

```sh
python3 -m unittest -v test_bridge.py
./sync preview
```

A failed push preserves the local commit; retry the same sync command after connectivity returns. A remote-ahead or divergent branch stops the sync without merging or force-pushing. Direct edits to exported copies stop the sync until reviewed. Common credential-pattern checks do not replace review of the export scope and content; errors identify a file and category without echoing the matched value.

After upgrading the script, run tests and review and commit implementation changes. Revert implementation commits through Git when needed. Never overwrite the vault from exported copies. To stop using the bridge, stop running its commands and move the external checkout away; Obsidian Sync is unaffected. No scheduled task or Obsidian Git plugin is installed.
