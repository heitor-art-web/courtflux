"""Pipeline performance profiler used by beta runs."""

from collections import defaultdict
from pathlib import Path
from time import perf_counter

import pandas as pd


class PipelineProfiler:
    def __init__(self):
        self.times = defaultdict(float)
        self.calls = defaultdict(int)
        self.starts = {}
        self.total_frames = 0
        self.skipped_frames = 0
        self.analysis_start = perf_counter()

    def start(self, section: str):
        self.starts[section] = perf_counter()

    def stop(self, section: str):
        start = self.starts.pop(section, None)
        if start is None:
            return
        elapsed = perf_counter() - start
        self.times[section] += elapsed
        self.calls[section] += 1

    def add(self, section: str, elapsed_seconds: float, calls: int = 1):
        self.times[section] += max(0.0, float(elapsed_seconds))
        self.calls[section] += max(0, int(calls))

    def frame_processed(self):
        self.total_frames += 1

    def frame_skipped(self):
        self.skipped_frames += 1
        self.total_frames += 1

    def snapshot(self, video_fps: float = 30.0) -> dict:
        runtime = perf_counter() - self.analysis_start
        analysed = max(0, self.total_frames - self.skipped_frames)
        analysis_fps = analysed / runtime if runtime else 0.0
        realtime = analysis_fps / video_fps if video_fps else 0.0

        stages = {}
        # Exclude Total when calculating the useful stage share because Total wraps
        # the other sections and would double-count their time.
        component_total = sum(v for k, v in self.times.items() if k != "Total")
        for stage, total in sorted(self.times.items(), key=lambda item: item[1], reverse=True):
            calls = self.calls[stage]
            stages[stage] = {
                "calls": calls,
                "total_sec": round(total, 4),
                "avg_ms": round(total / max(calls, 1) * 1000.0, 3),
                "component_pct": (
                    round(100.0 * total / component_total, 2)
                    if component_total and stage != "Total"
                    else None
                ),
                "wall_pct": round(100.0 * total / runtime, 2) if runtime else 0.0,
            }

        return {
            "wall_time_sec": round(runtime, 3),
            "frames_seen": self.total_frames,
            "frames_processed": analysed,
            "frames_skipped": self.skipped_frames,
            "analysis_fps": round(analysis_fps, 3),
            "video_fps": round(float(video_fps), 3),
            "realtime_factor": round(realtime, 3),
            "stages": stages,
        }

    def benchmark_dataframe(self):
        snap = self.snapshot()
        rows = []
        for stage, values in snap["stages"].items():
            rows.append(
                {
                    "Stage": stage,
                    "Calls": values["calls"],
                    "Total Seconds": values["total_sec"],
                    "Average ms": values["avg_ms"],
                    "Component %": values["component_pct"],
                    "Wall %": values["wall_pct"],
                }
            )
        return pd.DataFrame(rows)

    def save_csv(self, filename="benchmark.csv"):
        self.benchmark_dataframe().to_csv(filename, index=False)

    def report(self, video_fps=30.0, export_csv=False):
        snap = self.snapshot(video_fps=video_fps)
        print()
        print("=" * 76)
        print("Basketball AI Performance Report")
        print("=" * 76)
        print(f"Frames processed : {snap['frames_processed']}")
        print(f"Frames skipped   : {snap['frames_skipped']}")
        print(f"Wall time        : {snap['wall_time_sec']:.2f}s")
        print(f"Analysis FPS     : {snap['analysis_fps']:.2f}")
        print(f"Real-time factor : {snap['realtime_factor']:.2f}x")
        print("-" * 76)
        for stage, values in snap["stages"].items():
            component = values["component_pct"]
            component_text = "  n/a" if component is None else f"{component:5.1f}%"
            print(
                f"{stage:<28}{values['avg_ms']:>9.2f} ms   "
                f"{values['total_sec']:>9.2f}s   {component_text}"
            )
        print("=" * 76)

        if export_csv:
            output = Path("benchmark.csv")
            self.save_csv(output)
            print(f"Benchmark exported -> {output.resolve()}")
