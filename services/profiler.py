"""
Pipeline performance profiler.

Measures execution time for every pipeline stage and
generates a benchmark report.
"""

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

    # --------------------------------------------------

    def start(self, section: str):

        self.starts[section] = perf_counter()

    # --------------------------------------------------

    def stop(self, section: str):

        start = self.starts.pop(section, None)

        if start is None:
            return

        elapsed = perf_counter() - start

        self.times[section] += elapsed
        self.calls[section] += 1

    # --------------------------------------------------

    def frame_processed(self):

        self.total_frames += 1

    # --------------------------------------------------

    def frame_skipped(self):

        self.skipped_frames += 1

    # --------------------------------------------------

    def benchmark_dataframe(self):

        rows = []

        total_stage_time = sum(self.times.values())

        for stage in sorted(
            self.times,
            key=self.times.get,
            reverse=True,
        ):

            total = self.times[stage]

            rows.append({

                "Stage": stage,

                "Calls": self.calls[stage],

                "Total Seconds": round(total, 3),

                "Average ms":
                    round(
                        total /
                        max(self.calls[stage], 1)
                        * 1000,
                        2,
                    ),

                "Pipeline %":
                    round(
                        total /
                        total_stage_time
                        * 100,
                        1,
                    ) if total_stage_time else 0,

            })

        return pd.DataFrame(rows)

    # --------------------------------------------------

    def save_csv(
        self,
        filename="benchmark.csv",
    ):

        df = self.benchmark_dataframe()

        df.to_csv(
            filename,
            index=False,
        )

    # --------------------------------------------------

    def report(
        self,
        video_fps=30.0,
        export_csv=False,
    ):

        runtime = perf_counter() - self.analysis_start

        analysed = self.total_frames - self.skipped_frames

        analysis_fps = (
            analysed / runtime
            if runtime
            else 0
        )

        realtime = (
            analysis_fps / video_fps
            if video_fps
            else 0
        )

        total_stage_time = sum(self.times.values())

        print()

        print("=" * 70)
        print("Basketball AI Performance Report")
        print("=" * 70)

        print(f"Frames processed : {analysed}")
        print(f"Frames skipped   : {self.skipped_frames}")

        if self.total_frames:

            print(
                f"Skip Rate        : "
                f"{100*self.skipped_frames/self.total_frames:.1f}%"
            )

        print()

        print("-" * 70)
        print("Pipeline Breakdown")
        print("-" * 70)

        df = self.benchmark_dataframe()

        for _, row in df.iterrows():

            print(
                f"{row['Stage']:<24}"
                f"{row['Average ms']:>8.2f} ms"
                f"   "
                f"{row['Pipeline %']:>5.1f}%"
            )

        print()

        print("-" * 70)
        print("Performance")
        print("-" * 70)

        print(f"Video FPS         : {video_fps:.1f}")
        print(f"Analysis FPS      : {analysis_fps:.2f}")
        print(f"Real-Time Factor  : {realtime:.2f}x")

        print()

        print(f"Stage Time        : {total_stage_time:.2f}s")
        print(f"Wall Time         : {runtime:.2f}s")

        if runtime:

            overhead = runtime - total_stage_time

            print(f"Framework Overhead: {overhead:.2f}s")

        print("=" * 70)

        if export_csv:

            output = Path("benchmark.csv")

            self.save_csv(output)

            print()
            print(f"Benchmark exported -> {output.resolve()}")