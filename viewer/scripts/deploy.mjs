// The built site (dist/: the app and the data in public/) as the gh-pages branch, without piling up old data
// in the history (see below). Then publish it:
//   npm run deploy
//   git push -f origin gh-pages      (GitHub: Settings > Pages > deploy from branch gh-pages, / (root))
import { execFileSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const DIST = resolve(dirname(fileURLToPath(import.meta.url)), '../dist');
writeFileSync(join(DIST, '.nojekyll'), '');   // (serve the files as they are, no Jekyll)
const tmp = mkdtempSync(join(tmpdir(), 'deploy-'));
const GIT_DIR = execFileSync('git', ['rev-parse', '--absolute-git-dir'], { encoding: 'utf8' }).trim();
// (run in dist/ with the repository named explicitly: paths are dist/'s)
const git = (args, env = {}) => execFileSync('git', ['--git-dir', GIT_DIR, ...args],
  { cwd: DIST, encoding: 'utf8', env: { ...process.env, ...env }, maxBuffer: 1 << 26 }).trim();
try {
  // a throwaway index: the commit holds dist/ as the root, the working tree and the main index stay untouched
  const env = { GIT_INDEX_FILE: join(tmp, 'index') };
  git(['--work-tree', DIST, 'add', '--all', '--force', '.'], env);
  const tree = git(['write-tree'], env);
  const source = git(['rev-parse', '--short', 'HEAD']);
  // Git push only skips files GitHub already has through shared history. So normally the new site commit
  // builds on the previous one: the push sends just the changed files, and the history keeps the old
  // versions of what changed. When that would keep a lot (a new pipeline build: the tiles all differ, a full
  // upload anyway), it starts afresh, so old data never piles up in the history.
  const KEEP_MAX = 20e6;   // bytes of replaced or removed files the history may keep
  let previous = '';
  try { previous = git(['rev-parse', '--verify', '--quiet', 'refs/heads/gh-pages']); } catch { /* first deploy */ }
  let replaced = 0;
  if (previous) {
    const now = new Map(git(['ls-tree', '-r', tree]).split('\n').map((l) => [l.split('\t')[1], l.split('\t')[0].split(' ')[2]]));
    for (const l of git(['ls-tree', '-r', '-l', previous]).split('\n')) {
      const [meta, path] = l.split('\t'), [, , blob, size] = meta.split(/\s+/);
      if (now.get(path) !== blob) replaced += Number(size);
    }
  }
  const chain = previous !== '' && replaced < KEEP_MAX;
  const commit = git(['commit-tree', tree, ...(chain ? ['-p', previous] : []), '-m', `site built from ${source}`]);
  git(['update-ref', 'refs/heads/gh-pages', commit]);
  const files = git(['ls-tree', '-r', '--name-only', commit]).split('\n').length;
  if (!chain) {   // (the previous site and its data: no longer referenced, pruned locally)
    git(['reflog', 'expire', '--expire=now', 'refs/heads/gh-pages']);   // (only this branch's log)
    git(['gc', '--prune=now', '--quiet']);
  }
  console.log(chain ? `builds on the previous site: the push sends only what changed (${(replaced / 1e6).toFixed(1)} MB of files replaced)`
    : 'new data: the push sends the whole site');
  console.log(`gh-pages -> ${commit.slice(0, 7)} (built from ${source}, ${files} files). Publish with: git push -f origin gh-pages`);
} finally {
  rmSync(tmp, { recursive: true, force: true });
}
