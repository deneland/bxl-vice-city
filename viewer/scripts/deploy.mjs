// The built site (dist/: the app and the data in public/) as the gh-pages branch, one commit that replaces
// the previous one, so the data never piles up in the history. Then publish it:
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
  const commit = git(['commit-tree', tree, '-m', `site built from ${source}`]);
  git(['update-ref', 'refs/heads/gh-pages', commit]);
  const files = git(['ls-tree', '-r', '--name-only', commit]).split('\n').length;
  console.log(`gh-pages -> ${commit.slice(0, 7)} (built from ${source}, ${files} files). Publish with: git push -f origin gh-pages`);
} finally {
  rmSync(tmp, { recursive: true, force: true });
}
