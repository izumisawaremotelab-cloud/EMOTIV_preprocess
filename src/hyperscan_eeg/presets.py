"""ハードウェア・一般的なEEG研究慣習に基づく再利用可能なプリセット値。

特定の実験デザイン（被験者・条件・マーカー等）には依存しないが、
「EMOTIV Flex Salineを使う別の研究」や「標準的な周波数帯域を使う別の研究」
であれば共通して使い回せる値をここに集約する。実験固有の値は
`configs/` を参照。
"""

from __future__ import annotations

from .config import FrequencyBand

# EMOTIV Flex Saline 32ch のデジタイザ収録順（MATLAB由来の順序）。
EMOTIV_FLEX_SALINE_32CH_LABELS: tuple[str, ...] = (
    "C4", "T8", "FC2", "FC6", "FT10", "F4", "F8", "Fp2",
    "Oz", "Fz", "O2", "PO10", "P4", "P8", "CP2", "CP6",
    "C3", "T7", "FC1", "FC5", "FT9", "F3", "F7", "Fp1",
    "Pz", "Cz", "O1", "PO9", "P3", "P7", "CP1", "CP5",
)

# EMOTIV Flex Saline が出力する非EEG（補助）チャンネルの接頭辞。
EMOTIV_AUX_CHANNEL_PREFIXES: tuple[str, ...] = (
    "CQ.", "EQ.", "MOT.",
    "TimestampS", "TimestampMs", "OrTimestamp",
    "Counter", "Interpolated", "RawCq", "Battery", "BatteryPercent",
    "FwBuffer", "FwClock", "MarkerHardware",
)

# 一般的な5帯域（delta/theta/alpha/beta/gamma）。
STANDARD_EEG_FREQUENCY_BANDS: tuple[FrequencyBand, ...] = (
    FrequencyBand("delta", 1, 4),
    FrequencyBand("theta", 4, 8),
    FrequencyBand("alpha", 8, 13),
    FrequencyBand("beta", 13, 31),
    FrequencyBand("gamma", 31, 46),
)
