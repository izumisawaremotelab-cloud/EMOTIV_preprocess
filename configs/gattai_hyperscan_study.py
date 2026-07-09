"""この実験（2者間ハイパースキャニング・「合体」ゲーム課題）固有の設定。

`hyperscan_eeg` パッケージ本体（`src/hyperscan_eeg/`）は特定の実験デザインに
依存しない汎用ライブラリであり、被験者数・条件名・マーカーコード・
ペア対応・ゲーム状態などの具体的な値は一切持たない。本ファイルが、
それら実験固有の値を一箇所にまとめて注入する役割を持つ。

別の実験に転用する場合は、このファイルを複製・編集するだけでよく
（例: `configs/my_other_study.py`）、`src/hyperscan_eeg/` 側の変更は不要。
"""

from __future__ import annotations

from pathlib import Path

from hyperscan_eeg.config import (
    ExperimentDesign,
    MarkerConfig,
    MontageConfig,
    PathConfig,
    SegmentPlan,
)
from hyperscan_eeg.presets import EMOTIV_FLEX_SALINE_32CH_LABELS, STANDARD_EEG_FREQUENCY_BANDS

PATHS = PathConfig(
    raw_dir=Path("data/raw"),
    digitizer_dir=Path("data/digitizer"),
    preprocessed_dir=Path("data/preprocessed"),
    results_dir=Path("data/results"),
    figures_dir=Path("data/figures"),
)

# InletPortマーカー: このタスク制御ソフト固有の割り当て（2=タスク開始, 4=タスク終了）。
MARKER_CFG = MarkerConfig(start_marker=2, end_marker=4)

MONTAGE_CFG = MontageConfig(channel_labels=EMOTIV_FLEX_SALINE_32CH_LABELS)

FREQ_BANDS = STANDARD_EEG_FREQUENCY_BANDS

# 被験者3名(sub01~03)による総当たり3ペアで実施した実験のため、実施したペアを
# そのまま列挙する。総当たりでない実験構成では、以下を実施した組み合わせに
# 差し替えること（`ExperimentDesign.pairs` は常に「実際に実施したペア」を表す）。
DESIGN = ExperimentDesign(
    subjects=(1, 2, 3),
    conditions=("silent", "speaking"),
    pairs=((1, 2), (1, 3), (2, 3)),
    condition_segments={
        # 「合体」マーカー(3番)を境界として、silent条件は1回(2区間)、
        # speaking条件は2回(3区間)出現する、という実験固有の例外仕様。
        "silent": SegmentPlan(labels=("gattai_mae", "gattai_ato1"), boundary_markers=(3,)),
        "speaking": SegmentPlan(
            labels=("gattai_mae", "gattai_ato1", "gattai_ato2"), boundary_markers=(3, 3)
        ),
    },
)
