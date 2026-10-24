import json, math, statistics
from pathlib import Path
root = Path(__file__).resolve().parents[1]
data = json.loads((root/'hud-cost-final.json').read_text())
assert data['completed'] and data['identicalHealthyHTML'] and len(data['trials']) == 144

def stats(values):
    values = sorted(values)
    return dict(n=len(values), median=statistics.median(values), p95=values[math.ceil(.95*len(values))-1], mean=statistics.mean(values), variance=statistics.pvariance(values), maximum=max(values))

summary = []
for profile in ['fresh', '300', 'truncated']:
    for opened in [False, True]:
        trials = [t for t in data['trials'] if t['profile']==profile and t['open']==opened]
        arms = {version: [t for t in trials if t['version']==version] for version in ['before','after']}
        for arm in arms.values():
            assert len(arm) == 12 and sum(t['renders'] for t in arm) == 384 and sum(t['snapshots'] for t in arm) == 384
        row = dict(profile=profile, open=opened, arms={}, pairedMeanDeltas={})
        for version, arm in arms.items():
            row['arms'][version] = {stage: stats([v for t in arm for v in t['samples'][stage]]) for stage in ['snapshot','prepare','total']}
        for stage in ['snapshot','prepare','total']:
            differences=[]
            for pair in range(12):
                before=next(t for t in arms['before'] if t['pair']==pair)
                after=next(t for t in arms['after'] if t['pair']==pair)
                differences.append(statistics.mean(after['samples'][stage])-statistics.mean(before['samples'][stage]))
            row['pairedMeanDeltas'][stage]=stats(differences)
        summary.append(row)
(root/'hud-cost-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# PR06-F01 HUD preparation cost', '',
'The new validation/copy runs only at HUD publication, not per inference/capture/draw. This production-compiled browser comparison measures the changed boundary explicitly: old frozen PipelineDetails receives the real raw snapshot; corrected publication validates/copies the snapshot then renders the corrected memoized details. Both arms use the exact same unchanged PipelineMetrics implementation, React installation, CSS and deterministic valid input profile. The frozen before source differs only by its relocated metrics import. Healthy rendered HTML is identical after equalizing native disclosure state.', '',
'144 measured runs: six profiles (fresh/300/truncated × closed/open), 12 paired trials per profile, 32 real snapshot+React/DOM updates per trial, 384 updates per arm/profile, 2304 per arm total. Warmup16 per arm/profile excluded. Order alternates before/after then after/before (six pairs each ordering). A fresh snapshot object is created each update; no memoized no-op is timed. Actual snapshot/render counts match. The 300 profile populates all40 stage/outcome cells; the truncated profile sends2051 events per cell, retaining2048 with missing-duration entries and latest300 quantiles. This deliberately bounds the row reader while independently exercising unchanged expensive snapshot preparation.', '',
'All values are JS wall milliseconds. Median/p95 pool384 operation samples per arm/profile. Paired mean delta is after minus before, summarized over12 per-trial mean differences; nearest-rank p95 with12 pairs is the maximum, and variance is population ms². Zero due to clock quantization is retained. This is not CPU/GPU utilization, physical display latency or a device guarantee. Native disclosure state is kept equal, and no other test suite ran concurrently with the final sweep.', '',
'| Profile / disclosure | Prepare before median / p95 | Prepare corrected median / p95 | Paired prepare mean delta median / p95 / variance |', '| --- | --- | --- | --- |']
f=lambda v:f'{v:.5f}'
for r in summary:
    b=r['arms']['before']['prepare'];a=r['arms']['after']['prepare'];d=r['pairedMeanDeltas']['prepare']
    lines.append(f"| {r['profile']} / {'open' if r['open'] else 'closed'} | {f(b['median'])} / {f(b['p95'])} | {f(a['median'])} / {f(a['p95'])} | {f(d['median'])} / {f(d['p95'])} / {f(d['variance'])} |")
lines += ['', '| Profile / disclosure | Snapshot+prepare before median / p95 | Corrected median / p95 | Paired total mean delta median / p95 / variance |', '| --- | --- | --- | --- |']
for r in summary:
    b=r['arms']['before']['total'];a=r['arms']['after']['total'];d=r['pairedMeanDeltas']['total']
    lines.append(f"| {r['profile']} / {'open' if r['open'] else 'closed'} | {f(b['median'])} / {f(b['p95'])} | {f(a['median'])} / {f(a['p95'])} | {f(d['median'])} / {f(d['p95'])} / {f(d['variance'])} |")
lines += ['', '## Retention of earlier experiments', '',
'The collector/probe, scheduler, engine, Canvas renderer, existing FPS estimator and the actual image/SDK/ownership paths are byte-identical to reviewed04a3530; exact source bindings accompany this report. Their author288 and reviewer96 trials remain evidence at their original heads. Their old HUD/end-to-end values are not represented as measurements of the corrected HUD. This targeted experiment supplies the new changed-path measurement, so the long resolution/model pipeline sweeps were not repeated. No material change was made to measurement work per capture/comparison/SDK/draw or to source/processed/drawn counts; only the fixed40-row HUD reader is added at250ms publication. No arbitrary1ms acceptance threshold or improvement of the image algorithm is imposed.', '',
'Both positive/negative deltas, maxima, variance and raw samples remain in hud-cost-final.json/hud-cost-summary.json. Small timing changes, GC, browser scheduling and clock quantization limit attribution and tail precision. The full pressure profile is synthetic and is not a prediction for normal camera rates. The earlier native-open comparison mismatch and prior complete renderer-boundary sweeps are preserved in separate logs/JSON and are not substituted for this publication-boundary sweep.', '',
'Reproduce: node experiment/materialize-before.mjs from repository root; npx vite build --config docs/verification/pr06-f01-correction/experiment/vite.config.ts; serve.cache/pr06-f01-hud-build at127.0.0.1:4176; PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium node docs/verification/pr06-f01-correction/experiment/run-hud-cost.mjs; python3 docs/verification/pr06-f01-correction/experiment/summarize-hud.py.']
(root/'hud-cost-report.md').write_text('\n'.join(lines)+'\n')
print('Validated144 measured trials /4608 equal snapshot+render operations')
