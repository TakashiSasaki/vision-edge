import json, statistics, math
from pathlib import Path
root = Path(__file__).resolve().parents[1]
data = json.loads((root / 'overhead.json').read_text())
def describe(xs):
    xs = sorted(xs)
    return dict(n=len(xs), median=statistics.median(xs), p95=xs[math.ceil(.95 * len(xs)) - 1], mean=statistics.mean(xs), variance=statistics.pvariance(xs), maximum=max(xs))
configurations = sorted({(t['width'], t['height'], t['path'], t['model']) for t in data['trials']})
summary = []
for width, height, path, model in configurations:
    selected = [t for t in data['trials'] if (t['width'], t['height'], t['path'], t['model']) == (width, height, path, model)]
    arms = {enabled: [t for t in selected if t['enabled'] == enabled] for enabled in (False, True)}
    for enabled, trials in arms.items():
        for t in trials:
            n = t['iterations']; c = t['counts']
            assert (c['processed'], c['sdk'], c['drawn'], c['captured'], c['compared']) == (n, n, n, 3*n, 5*n-3 if path == 'rgba' else 0)
            if not enabled: assert c['metricClock'] == c['snapshot'] == 0
            assert t['checksum'] == n
    row = dict(width=width, height=height, path=path, model=model, trialsPerArm=len(arms[False]), arms={}, pairedDeltas={})
    for enabled, trials in arms.items():
        key = 'enabled' if enabled else 'disabled'
        row['arms'][key] = dict(counts={k: sum(t['counts'][k] for t in trials) for k in trials[0]['counts']}, statistics={})
        for stage in ('total', 'engine', 'draw', 'hud'):
            row['arms'][key]['statistics'][stage] = describe([v for t in trials for v in t['samples'][stage]])
    by_pair = {pair: {t['enabled']: t for t in selected if t['pair'] == pair} for pair in range(len(arms[False]))}
    for stage in ('total', 'engine', 'draw', 'hud'):
        row['pairedDeltas'][stage] = describe([pair[True][stage]['mean'] - pair[False][stage]['mean'] for pair in by_pair.values()])
    summary.append(row)
(root / 'overhead-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
lines = ['# PR06 matched-work synthetic overhead', '',
'Measurement implementation built at `c4ce53557e0e74798ff69dddf529ba2c5a447ad6`; final product `9865050bd1ee025ecc3e39021dfcd103b6f7acc7` changes only the fatal-fault snapshot reset and its test. The benchmark imports no VisionCanvas module; all benchmark-imported product modules are byte-identical across these commits, and the rebuilt bundle hash is checked. Same production-compiled experiment bundle for both arms. Raw per-operation samples and all trials: [overhead.json](overhead.json); exact aggregate/count validation: [overhead-summary.json](overhead-summary.json).', '',
'12 paired trials per resolution/path/model (24 measured runs); 16 processed jobs per RGBA run and 32 per VideoFrame run. Each configuration warms both arms first (8/16 jobs), excluded from statistics. Pair order alternates disabled/enabled then enabled/disabled, six of each ordering. Each job changes only the final source pixel, then polls a duplicate. The full-array scan callback and dimensions/bytes algorithm are unchanged. Actual production-boundary work counts must match in both arms or the runner fails. The comparison counter measures actual TypedArray.every full-array scans in both arms; nullable samePixels invocations also include cheap checks and have a different count. No lifetime count is inferred from five-second snapshot retention.', '',
'For RGBA each arm/configuration processes 192 jobs, captures 576 times, performs 924 full-array comparison scans and copies 192 results. For VideoFrame each arm/configuration processes/copies 384 jobs and captures/releases 1152 frames, with no pixel comparisons. The raw counts.released counter covers VideoFrame.close only; RGBA Canvas release is verified by ownership tests and is not counted by that VideoFrame-specific counter. Both model doubles use the same scripted images; the segmenter supplies a uniform full-input-resolution mask, deliberately synthetic. Snapshot/quantile computation plus actual React/DOM detailed-panel preparation executes once per eight synthetic 30fps source frames (approximately the app 250 ms publication cadence). Disabled runs have zero instrumentation clock reads, zero event/identity allocation through the optional probe paths, and zero snapshots/panel updates. Shared external stopwatch/counter overhead applies to both arms. Work is synchronous and timer callbacks are manually driven to avoid timer-clamp noise; native queue delay is covered separately by deterministic scheduler and real-app browser tests.', '',
'Numbers below are **external observed milliseconds**. Median and nearest-rank p95 pool all per-operation samples (192 RGBA or 384 VideoFrame per arm); paired deltas are enabled-minus-disabled **per-trial means**, with median, p95 and population variance (ms²) across 12 pairs. With 12 pairs the nearest-rank p95 of paired mean deltas is the maximum pair. Engine includes production wrapper/probes. Draw includes shared real Canvas renderer plus enabled draw/first-draw bookkeeping. Total includes capture/comparison/engine/draw and the periodic snapshot/React/DOM cost. HUD is snapshot plus detailed-panel preparation, with 24/48 updates per enabled arm. Disabled HUD timings are external stopwatch measurements of the bypass branch, not 0 ms snapshot executions; the actual snapshot count is zero. The synthetic every-eight-frame publication schedule is explicit and is not a measured real-time 4 Hz scheduler under slow RGBA work; separate HUD costs are provided so the weighting is visible. This harness does not reproduce every VisionCanvas overlay label/ripple, React App notification, browser RAF gap, physical camera or model/GPU operation. It is a local instrumentation-overhead experiment, not a device guarantee or speedup claim.', '',
'| Resolution / path / model | Total disabled median / p95 | Total enabled median / p95 | Paired total mean delta median / p95 / variance |',
'| --- | --- | --- | --- |']
f = lambda v: f'{v:.4f}'
for r in summary:
    off=r['arms']['disabled']['statistics']['total']; on=r['arms']['enabled']['statistics']['total']; delta=r['pairedDeltas']['total']
    lines.append(f"| {r['width']}×{r['height']} / {r['path']} / {r['model']} | {f(off['median'])} / {f(off['p95'])} | {f(on['median'])} / {f(on['p95'])} | {f(delta['median'])} / {f(delta['p95'])} / {f(delta['variance'])} |")
for stage, label in [('engine', 'Engine'), ('draw', 'Canvas draw and first-draw bookkeeping'), ('hud', 'Snapshot, quantiles and React/DOM preparation')]:
    lines += ['', f'## {label}', '', '| Resolution / path / model | Disabled median / p95 | Enabled median / p95 | Paired mean delta median / p95 / variance |', '| --- | --- | --- | --- |']
    for r in summary:
        off=r['arms']['disabled']['statistics'][stage]; on=r['arms']['enabled']['statistics'][stage]; delta=r['pairedDeltas'][stage]
        lines.append(f"| {r['width']}×{r['height']} / {r['path']} / {r['model']} | {f(off['median'])} / {f(off['p95'])} | {f(on['median'])} / {f(on['p95'])} | {f(delta['median'])} / {f(delta['p95'])} / {f(delta['variance'])} |")
lines += ['', '## Interpretation and retained limits', '',
'Clock resolution creates many legitimate 0 ms intervals for the cheap deterministic SDK. Added instrumentation is measurable in the cheaper paths and in HUD preparation; percentages relative to a near-zero SDK are unstable. Positive and negative paired deltas and outliers are preserved. Negative values indicate local noise/order/GC variation and are not evidence that instrumentation optimizes inference. The unchanged full-resolution RGBA identity scans dominate the slower paths. No arbitrary 1 ms threshold or universal acceptance budget is imposed. The implementation keeps fixed event rings, a separate latest-300 quantile cap, fixed job retention, no RAF sorting, and 250 ms snapshots; these are instrumentation design bounds, not an image-algorithm optimization.', '',
'`overhead-initial-partial.json` preserves the original harness failure when rolling counts were mistaken for lifetime counts. `overhead-pre-occupancy.json` is the complete earlier diagnostic sweep before the final job-occupancy probe, with concurrent author validation during some trials; it is not substituted for this final-source sweep. Its unfavorable observations remain. The bulk browser/controls suites completed before this sweep. A 2.3-second focused fault-UI regression ran at 15:07:26 UTC during the sweep; this possible external noise is disclosed and its observations are retained. No effect is causally attributed to it. System Chromium and cloud CPU results are distinct from hosted matched-browser CI and from untested physical devices. Raw environment/browser version and warmup data are included in overhead.json.', '',
'Reproduce: `npx vite build --config docs/verification/pr06/experiment/vite.overhead.config.ts`; serve `.cache/pr06-overhead` at 127.0.0.1:4175; `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium node docs/verification/pr06/experiment/run-overhead.mjs`; `python3 docs/verification/pr06/experiment/summarize-overhead.py`.']
(root / 'overhead-report.md').write_text('\n'.join(lines)+'\n')
print(f'Validated {len(data["trials"])} trials in {len(summary)} configurations')
