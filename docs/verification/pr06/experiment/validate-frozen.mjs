import { spawnSync, execFileSync } from 'node:child_process';
import { writeFileSync } from 'node:fs';
const commands = [
  ['npm', ['run', 'typecheck']], ['npm', ['run', 'lint']], ['npm', ['test']], ['npm', ['run', 'build']],
  ['npm', ['test', '--', '--run', 'src/services/pipelineMetrics.test.ts', 'src/services/frameScheduler.test.ts', 'src/services/visionEngine.metrics.test.ts', 'src/components/VisionCanvas.test.tsx']],
  ['git', ['diff', '--check']],
];
const receipt = { head: execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim(), commands: [] };
for (const [index, [command, args]] of commands.entries()) {
  const start = new Date().toISOString(); const result = spawnSync(command, args, { encoding: 'utf8' });
  const log = `docs/verification/pr06/logs/validated-${receipt.head.slice(0, 7)}-${index}-${command}-${args[0]}.log`;
  writeFileSync(log, result.stdout + result.stderr);
  receipt.commands.push({ command: [command, ...args], start, end: new Date().toISOString(), exit: result.status, signal: result.signal, log });
  writeFileSync('docs/verification/pr06/final-commands.json', JSON.stringify(receipt, null, 2) + '\n');
  console.log(`${command} ${args.join(' ')}: exit=${result.status}`);
  if (result.status !== 0) process.exit(result.status ?? 1);
}
