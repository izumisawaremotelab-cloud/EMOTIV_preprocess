"""脳間同期指標（PLV / PSI 等）の算出。

旧実装は変数・ファイル名は「PSI」だが `method='plv'` で計算しており、
実際にはPhase Locking Value（位相同期値）を算出していた。PSI
（Phase Slope Index、位相スロープ指数）とPLVは定義の異なる指標であるため、
この不一致はバグとして扱い、`method` を明示的に指定する設計に改めた
（詳細はモジュールの解説を参照）。

補足（重要）: PLV等のエポック平均を前提とする指標は、1エポックしか
与えないと常に自明な値になる。`mne_connectivity` の `_PLVEst.accumulate`
は各エポックについて単位modulusの複素数 `csd_xy / |csd_xy|` を加算し、
`compute_con` で `n_epochs` 個の平均の絶対値を取るだけなので、
`n_epochs=1` なら `|csd_xy / |csd_xy|| = 1` が定義上常に成り立つ
（＝ヒートマップが全部1になるバグの直接原因）。本モジュールは連続データ
（`phase.combine_dyad_raw` が返す1つながりのRaw）を対象にするため、
`compute_band_connectivity` 内で毎回、固定長・非重複の疑似エポックに
分割してから `spectral_connectivity_epochs` に渡す。
"""

from __future__ import annotations

import logging

import mne
import numpy as np
import pandas as pd
from mne_connectivity import spectral_connectivity_epochs

from .config import FrequencyBand
from .phase import make_pseudo_trial_epochs

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

# 疑似エポック長を決める際の既定値。
#
# `ideal_epoch_length = 2 * worst_edge + _CORE_SECONDS` の形で、
# ウェーブレットの片側エッジ効果幅(worst_edge)を両側分確保した上に、
# 歪みのない「有効な中心区間」を _CORE_SECONDS 秒だけ上乗せする。
# 区間長が短くこの理想値では _MIN_EPOCHS 個のエポックを確保できない場合は
# `duration / _MIN_EPOCHS` まで縮小する（縮小時は必ず警告ログを出す）。
_CORE_SECONDS = 2.0
_MIN_EPOCHS = 3


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


def _determine_epoch_length(
    freqs: np.ndarray,
    cwt_n_cycles: np.ndarray,
    duration: float,
    core_seconds: float,
    min_epochs: int,
    method: str,
) -> float:
    """帯域の周波数構成とウェーブレットのエッジ効果から疑似エポック長を決める。

    Raises:
        ValueError: 縮小してもなお `min_epochs` に満たない疑似エポックしか
            作れないほど区間が短い場合。
    """
    worst_edge = np.max(cwt_n_cycles / freqs)  # 片側のエッジ効果幅（秒）
    ideal_epoch_length = 2 * worst_edge + core_seconds

    epoch_length = ideal_epoch_length
    if duration / epoch_length < min_epochs:
        epoch_length = duration / min_epochs
        logger.warning(
            "区間長(%.1fs)に対し理想的な疑似エポック長(%.1fs、片側エッジ効果幅%.1fs)"
            "では%d個のエポックを確保できないため、%.1fsに縮小します"
            "（低周波数ビンの推定精度が低下する可能性があります）。",
            duration, ideal_epoch_length, worst_edge, min_epochs, epoch_length,
        )

    min_cycle_duration = 1.0 / np.min(freqs)
    if epoch_length < min_cycle_duration or duration / epoch_length < 2:
        raise ValueError(
            f"区間長({duration:.1f}s)が短すぎるため、'{method}'の算出に必要な"
            f"疑似エポック(最低2個、各{min_cycle_duration:.1f}s以上)を作成できません。"
        )
    return epoch_length


def compute_band_connectivity(
    combined_raw: mne.io.BaseRaw,
    sfreq: float,
    freqs: np.ndarray,
    method: str = "plv",
    suffix_a: str = "_p1",
    suffix_b: str = "_p2",
    cwt_n_cycles: np.ndarray | float | None = None,
    core_seconds: float = _CORE_SECONDS,
    min_epochs: int = _MIN_EPOCHS,
) -> pd.DataFrame:
    """指定した周波数帯域における、電極ペア全組み合わせの脳間同期指標を算出する。

    Args:
        combined_raw: `phase.combine_dyad_raw` で結合した連続Raw。
        sfreq: サンプリング周波数。
        freqs: 解析対象の周波数ビン配列。
        method: `mne_connectivity.spectral_connectivity_epochs` の method
            （例: 'plv', 'psi', 'coh', 'wpli'）。
        cwt_n_cycles: Morletウェーブレットのサイクル数。None の場合、
            `freqs / 2` を [3, 15] にクリップして自動決定する。
        core_seconds: 疑似エポック長を決める際に、両側エッジ効果幅の上に
            上乗せする「歪みのない中心区間」の長さ（秒）。
        min_epochs: 確保したい疑似エポック数の下限。区間が短くこれを
            満たせない場合は疑似エポック長を縮小する。

    Returns:
        p1側電極(行) x p2側電極(列)の32x32相当のDataFrame（帯域平均済み）。
    """
    idx_a, idx_b, pair_names = _paired_channel_indices(
        combined_raw.ch_names, suffix_a, suffix_b
    )
    n = len(idx_a)

    if cwt_n_cycles is None:
        cwt_n_cycles = np.clip(freqs / 2.0, _MIN_CYCLES, _MAX_CYCLES)

    duration = combined_raw.times[-1]
    epoch_length = _determine_epoch_length(
        freqs, cwt_n_cycles, duration, core_seconds, min_epochs, method
    )
    epochs = make_pseudo_trial_epochs(combined_raw, epoch_length)
    logger.info(
        "疑似エポック数: %d個（各%.1fs、区間長%.1fs）", len(epochs), epoch_length, duration
    )

    seed_idx = np.repeat(idx_a, n)
    target_idx = np.tile(idx_b, n)

    con = spectral_connectivity_epochs(
        epochs,
        method=method,
        mode="cwt_morlet",
        cwt_freqs=freqs,
        cwt_n_cycles=cwt_n_cycles,
        indices=(seed_idx, target_idx),
        sfreq=sfreq,
        verbose=False,
    )

    # 形状 (Shape): (電極ペア数=n*n, 周波数数[, 時間数])。時間軸があればさらに平均する。
    raw_outputs = con.get_data()
    band_averaged = np.mean(raw_outputs, axis=tuple(range(1, raw_outputs.ndim)))
    matrix = band_averaged.reshape(n, n)

    index = pd.Index(pair_names, name="ch_p1")
    columns = pd.Index(pair_names, name="ch_p2")
    return pd.DataFrame(matrix, index=index, columns=columns)


def compute_all_bands(
    combined_raw: mne.io.BaseRaw,
    sfreq: float,
    bands: tuple[FrequencyBand, ...],
    method: str = "plv",
    suffix_a: str = "_p1",
    suffix_b: str = "_p2",
) -> dict[str, pd.DataFrame]:
    """複数の周波数帯域について同期指標をまとめて算出する。

    Returns:
        帯域名 -> (p1側電極 x p2側電極)の同期指標行列 のdict。
    """
    results: dict[str, pd.DataFrame] = {}
    for band in bands:
        logger.info("計算中: %s帯域 (%s)", band.name, method)
        results[band.name] = compute_band_connectivity(
            combined_raw, sfreq, band.freqs(), method=method,
            suffix_a=suffix_a, suffix_b=suffix_b,
        )
    return results
