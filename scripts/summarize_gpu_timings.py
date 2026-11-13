"""Summarize viewer CPU/GPU stage timings, excluding warm-up frames."""
import argparse
import csv
import json
import math
import io
from pathlib import Path
import statistics

STAGES = ('scene_update_host_ms', 'neeat_gpu_ms', 'path_trace_gpu_ms',
          'denoise_total_gpu_ms', 'dlss_evaluate_gpu_ms', 'film_convert_gpu_ms',
          'display_gpu_ms', 'worker_to_gpu_display_ms', 'observed_ms',
          'present_backpressure_ms', 'gpu_queue_wait_host_ms', 'display_reader_wait_host_ms')


def distribution(values):
    values = sorted(values)
    if not values:
        return {'samples': 0}
    index = .95 * (len(values) - 1)
    lo = int(index)
    p95 = values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (index - lo)
    return dict(samples=len(values), mean_ms=statistics.mean(values),
                p50_ms=statistics.median(values), p95_ms=p95)


def summarize(rows, warmup):
    groups = {}
    by_frame = {int(row['frame']): row for row in rows}
    seen_display = set()
    previous = None
    for row in rows:
        frame = int(row['frame'])
        key = row['mode'] + '/' + row.get('phase', 'unknown')
        if frame >= warmup:
            group = groups.setdefault(key, {stage: [] for stage in STAGES + ('present_interval_ms',)})
            for stage in STAGES:
                # Display timestamps belong to the explicitly reported earlier
                # frame, not necessarily the row carrying the completed query.
                if stage == 'display_gpu_ms':
                    source = int(row.get('display_gpu_frame') or -1)
                    if source < warmup:
                        continue
                    owner = by_frame.get(source)
                    if owner is None or owner['mode'] != row['mode'] or owner.get('phase') != row.get('phase'):
                        continue
                    if source in seen_display:
                        continue
                    seen_display.add(source)
                if row.get(stage, ''):
                    value = float(row[stage])
                    if math.isfinite(value) and value >= 0:
                        group[stage].append(value)
            if previous and int(previous['frame']) + 1 == frame and previous['mode'] == row['mode'] and previous.get('phase') == row.get('phase') and int(previous['frame']) >= warmup:
                interval = float(row['present_timestamp_ms']) - float(previous['present_timestamp_ms'])
                if interval > 0:
                    group['present_interval_ms'].append(interval)
        previous = row
    return {key: {stage: distribution(values) for stage, values in group.items()}
            for key, group in groups.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--warmup', type=int, default=32)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error('warmup must be nonnegative')
    snapshot = args.csv.read_text(encoding='utf-8')
    # A running viewer may have flushed only part of the final CSV record.
    if snapshot and not snapshot.endswith('\n'):
        snapshot = snapshot.rsplit('\n', 1)[0] + '\n'
    rows = list(csv.DictReader(io.StringIO(snapshot)))
    result = dict(source=str(args.csv), warmup=args.warmup,
                  groups=summarize(rows, args.warmup),
                  notes='GPU values are event timeline spans, not summed kernel busy time. DLSS evaluate is nested in denoise total. Display uses its own frame ID. Do not sum nested stages or CPU/GPU clocks.')
    output = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(output, encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
