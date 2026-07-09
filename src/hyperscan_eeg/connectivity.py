"""脳間同期指標（PLV / PSI 等）の算出。

旧実装は変数・ファイル名は「PSI」だが `method='plv'` で計算しており、
実際にはPhase Locking Value（位相同期値）を算出していた。PSI
（Phase Slope Index、位相スロープ指数）とPLVは定義の異なる指標であるため、
この不一致はバグとして扱い、`method` を明示的に指定する設計に改めた
（詳細はモジュールの解説を参照）。
"""

from __future__ import annotations

import logging

import mne
import numpy as np
import pandas as pd
from mne_connectivity import spectral_connectivity_epochs

from .config import FrequencyBand

logger = logging.getLogger(__name__)

# cwt_n_cycles を周波数に比例させて自動決定する際の下限・上限。
#
# 根拠: 「n_cycles = freqs / 2」は、全周波数で時間窓長を一定(0.5秒)に揃える
# 目的でMNE公式ドキュメントが例示している式であり(参照:
# https://mne.tools/stable/generated/mne.time_frequency.morlet.html の
# Notes、"To use a temporal window with fixed length..." の記述)、独自に
# 考案した式ではない。一方 [3, 15] のクリップ範囲そのものは特定の論文や
# MNEの推奨値に基づくものではなく、以下の実務上の理由による本実装独自の
# 安全策:
#   - 下限3: freqs/2 をそのまま使うとdelta帯域(1-4Hz)でn_cycles<1となり、
#     ウェーブレットとして周期性が定義できず位相推定が不安定になる
#     （旧実装の固定値 n_cycles=7 は逆にdelta帯で片側3.5秒ものエッジ効果を
#     生み、本実験の短い区間(約13秒)では解析可能な区間の大半を侵食していた）。
#   - 上限15: 高周波数帯で不必要にウェーブレットを長くしないための上限。
# 文献に基づく具体的な値を使いたい場合は、`compute_band_connectivity` の
# `cwt_n_cycles` に明示的な配列を渡せば本ヒューリスティックは使われない。
_MIN_CYCLES = 3.0
_MAX_CYCLES = 15.0


def _paired_channel_indices(
    ch_names: list[str], suffix_a: str, suffix_b: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """`_p1` / `_p2` 等のサフィックスを手がかりに、対応する電極ペアの
    インデックスを名前ベースで安全に決定する。

    旧実装はチャンネル順序が2者間で一致していることを無条件に仮定していた
    （`indices = (arange(n_ch), arange(n_ch)+n_ch)`）。ここではベース名の
    並びが完全一致することを明示的に検証し、不一致なら例外を送出する。
    """
    names_a = [ch[: -len(suffix_a)] for ch in ch_names if ch.endswith(suffix_a)]
    names_b = [ch[: -len(suffix_b)] for ch in ch_names if ch.endswith(suffix_b)]
    if names_a != names_b:
        raise ValueError(
            "2者間のチャンネル構成/順序が一致しないため電極ペアを決定できません。"
            f"p1側: {names_a}\np2側: {names_b}"
        )
    idx_a = np.array([ch_names.index(f"{n}{suffix_a}") for n in names_a])
    idx_b = np.array([ch_names.index(f"{n}{suffix_b}") for n in names_b])
    return idx_a, idx_b, names_a


def compute_band_connectivity(
    combined_epochs: mne.Epochs,
    sfreq: float,
    freqs: np.ndarray,
    method: str = "plv",
    suffix_a: str = "_p1",
    suffix_b: str = "_p2",
    cwt_n_cycles: np.ndarray | float | None = None,
) -> tuple[np.ndarray, list[str]]:
    """指定した周波数帯域における帯域平均済みの脳間同期指標を算出する。

    Args:
        combined_epochs: `phase.combine_dyad_epochs` で結合したEpochs。
        sfreq: サンプリング周波数。
        freqs: 解析対象の周波数ビン配列。
        method: `mne_connectivity.spectral_connectivity_epochs` の method
            （例: 'plv', 'psi', 'coh', 'wpli'）。
        cwt_n_cycles: Morletウェーブレットのサイクル数。None の場合、
            `freqs / 2` を [3, 15] にクリップして自動決定する。

    Returns:
        (電極ペアごとの帯域平均値, 電極ペア名リスト) のタプル。
    """
    idx_a, idx_b, pair_names = _paired_channel_indices(
        combined_epochs.ch_names, suffix_a, suffix_b
    )

    if cwt_n_cycles is None:
        cwt_n_cycles = np.clip(freqs / 2.0, _MIN_CYCLES, _MAX_CYCLES)
        duration = combined_epochs.tmax - combined_epochs.tmin
        worst_edge = np.max(cwt_n_cycles / freqs)  # 片側のエッジ効果幅（秒）
        if 2 * worst_edge > duration:
            logger.warning(
                "区間長(%.1fs)に対しウェーブレットのエッジ効果幅(%.1fs x2)が大きく、"
                "低周波数ビンの推定精度が低下する可能性があります。",
                duration, worst_edge,
            )

    con = spectral_connectivity_epochs(
        combined_epochs,
        method=method,
        mode="cwt_morlet",
        cwt_freqs=freqs,
        cwt_n_cycles=cwt_n_cycles,
        indices=(idx_a, idx_b),
        sfreq=sfreq,
        verbose=False,
    )

    # 形状 (Shape): (ペア数, 周波数数[, 時間数])。時間軸がある場合はさらに平均する。
    raw_outputs = con.get_data()
    band_averaged = np.mean(raw_outputs, axis=tuple(range(1, raw_outputs.ndim)))
    return band_averaged, pair_names


def compute_all_bands(
    combined_epochs: mne.Epochs,
    sfreq: float,
    bands: tuple[FrequencyBand, ...],
    method: str = "plv",
    suffix_a: str = "_p1",
    suffix_b: str = "_p2",
) -> pd.DataFrame:
    """複数の周波数帯域について同期指標をまとめて算出し、DataFrameで返す。"""
    columns: dict[str, np.ndarray] = {}
    pair_labels: list[str] | None = None

    for band in bands:
        logger.info("計算中: %s帯域 (%s)", band.name, method)
        values, pair_names = compute_band_connectivity(
            combined_epochs, sfreq, band.freqs(), method=method,
            suffix_a=suffix_a, suffix_b=suffix_b,
        )
        if pair_labels is None:
            pair_labels = [f"{n}{suffix_a} <-> {n}{suffix_b}" for n in pair_names]
        columns[band.name] = values

    return pd.DataFrame(columns, index=pair_labels)
