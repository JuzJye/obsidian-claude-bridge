# Selected Obsidian knowledge for Claude

Start with [Home](Home.md). This public repository contains explicitly reviewed copies of learning notes, exercises, paper notes, general tool notes, reusable methods, a concise work handoff, and the figures/small results they reference. Obsidian is the source of truth. This repository is a publication subset, not a full vault backup.

Anyone can read published files and Git history. Original company documents, machine operations, complete personal memory and conversations, the inbox, external skill packages, and application state stay outside this repository.

## Sync

Ask local Codex to sync the selected knowledge, or run these commands from the external checkout:

```sh
./sync preview
./sync preview --summary
./sync sync
```

`preview` is read-only and offline. It lists the included source files, excluded/unselected inventory, excluded links, missing references, sanitized text differences, and binary hashes. `--summary` hides the long file listing and full diffs. Source content is validated before any diff is printed, and matched secrets are never echoed.

`sync` verifies remote visibility, checkout state, and pending commit history, exports the reviewed payload, commits changed files, and pushes. Repeating it without source changes does not create a new commit. A failed push preserves the local commit for retry. A remote-ahead or divergent branch stops without a merge or force push.

## Reading from Claude

Use English for requests and collaboration instructions. One short request is enough:

> Read this index and the relevant linked notes: https://raw.githubusercontent.com/JuzJye/obsidian-claude-bridge/main/Home.md

Home groups the Markdown notes by topic and provides direct Raw URLs. Relative links in raw notes resolve against the same raw URL base. Exercises, figures, and selected JSON/CSV evidence keep their relative paths. Code-block and inline-code link examples are left as examples. Unsupported, excluded, and missing navigation references are explicitly marked rather than silently publishing their targets. No placeholders are generated for unavailable files.

After a push, ask Claude to fetch updated files again and verify the export timestamp. A commit-specific raw URL can identify an exact revision. GitHub's directory web page may be unavailable to a chat's fetch tool even while public raw-text files are readable. Selecting a repository in a picker alone does not establish actual retrieval capability.

Claude feedback is reviewed by local Codex before updating the original note. Chat replies are not automatically written back or dispatched as tasks. Useful clipped excerpts remain in the local inbox until reviewed and explicitly selected for publication.

## Publication policy

`export.json` contains an explicit `files` list and additional `exclude` patterns. New files are not automatically published because they were added to a directory. Built-in exclusions take precedence even over explicit allowlist entries, and configuration can add restrictions but cannot remove the built-in restrictions.

Mandatory exclusions cover external skills, local Claude network/deployment records and diagnostic attachments, hidden application state, raw conversations and private project/memory records, the inbox and daily/private notes, workplace runbooks/reports, local Linux connection notes, project-specific skill cases, caches, logs, weights, NPZ arrays, audio, and original paper PDFs. The sole reviewed project export is the curated work handoff. Original paper source links are retained.

Supported selected files are Markdown, Python, JSON, CSV, SVG, PNG, JPEG, and WebP. Markdown export copies may receive explicitly configured local substitutions, removal of reviewed private-context lines, and link adjustments. Python and supporting evidence are copied byte-for-byte and hashed; unresolved credentials or machine-specific information stop export rather than silently editing those files. Text/evidence files are limited to 1 MiB and binary files to 10 MiB. Source files must be regular, non-symlink files inside the vault.

To expand the selection, review the exact source files, their required references, and any substitutions, update the explicit list, run preview, then review and commit configuration changes separately before sync. The bridge does not automatically traverse links or publish company summaries. Reducing the list or losing a selected source file stops instead of automatically withdrawing old public files. Removing a published file does not erase Git history.

## Local setup

Requires macOS/Linux, Python 3.9+, Git, and an authenticated GitHub CLI for pushing. Keep the checkout outside the vault, on one designated bridge computer. No scheduled sync, Obsidian Sync setting change, or Git initialization in the vault is required.

Create the ignored `.bridge-local.json` on that computer:

```json
{
  "vault": "/absolute/path/to/vault",
  "replacements": [["internal.example", "gitlab.example.com"]],
  "markdown_omit_lines": {}
}
```

`replacements` are literal substitutions in Markdown export copies only. `markdown_omit_lines` maps a selected Markdown path to reviewed regular-expression removals of local-only context. Keep these local rules out of Git. Recreate and review them on a new computer before publication; an unhandled internal domain, private network address, or machine-specific user path stops the export.

Authentication uses GitHub CLI/system credential storage. Never put tokens in notes or configuration. The remote must match the configured GitHub repository. The explicit visibility here is `public`; missing visibility defaults to `private` for compatibility, and a mismatch stops sync.

## Verification and recovery

```sh
python3 -m unittest -v test_bridge.py
```

Tests use isolated source vaults and local bare remotes. They cover selection and privacy boundaries, exact attachment copying, visibility mismatches, credential checks, retries, and remote-ahead/divergent histories. Checks of common credential formats supplement a content review; they do not guarantee detection of every sensitive fact.

Direct edits to export copies or an existing staged change stop routine sync. Review them before proceeding. Update implementation files separately and run tests before committing them. Use Git revert for implementation rollback. Never restore source notes by blindly copying public export files over the vault.

To stop using the bridge, stop running the commands and move the external checkout away. Obsidian Sync is unaffected. Local Claude operations and backups are maintained outside the publication list.
