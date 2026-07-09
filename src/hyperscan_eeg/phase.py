"""2者間ダイアド解析のための位相計算前処理（エポック結合）。

`prepare_whole_task_epochs`（旧実装）は「タスク全体」専用だったが、
ここでは任意の2つのRaw区間（タスク全体でも、ゲーム状態ごとの
サブ区間でも）を受け取れる汎用関数として実装する。

補足: このエポック結合はPSI固有の要件ではない。
`mne_connectivity.spectral_connectivity_epochs` はPLV/PSI/coherence等
どの指標を使う場合でも「1つのEpochs/Rawオブジェクト内のチャンネル同士」の
接続性しか計算できないAPI仕様であるため、2者を比較するには両者の
チャンネルを1つのオブジェクトへ合体させる必要がある。ここでの
「1エポック化」も、連続データをEpochsコンテナに載せるための便宜的な
ラップであり、複数試行に分けた古典的エポック化ではない。
"""

from __future__ import annotations

import logging

import mne
import numpy as np

logger = logging.getLogger(__name__)


def combine_dyad_epochs(
    raw_a: mne.io.BaseRaw,
    raw_b: mne.io.BaseRaw,
    suffix_a: str = "_p1",
    suffix_b: str = "_p2",
) -> tuple[mne.Epochs, float]:
    """2者分のRaw区間全体を1つの巨大エポックとして結合する。

    双方のチャンネル名が衝突しないよう `suffix_a` / `suffix_b` を付与した上で
    1つのEpochsオブジェクトにまとめる。区間長が異なる場合は短い方に合わせる。

    Raises:
        ValueError: サンプリング周波数が一致しない場合。
    """
    sfreq_a = raw_a.info["sfreq"]
    sfreq_b = raw_b.info["sfreq"]
    if sfreq_a != sfreq_b:
        raise ValueError(f"サンプリング周波数が一致しません: {sfreq_a} Hz vs {sfreq_b} Hz")
    sfreq = sfreq_a

    events_a = np.array([[raw_a.first_samp, 0, 1]])
    events_b = np.array([[raw_b.first_samp, 0, 1]])

    min_samples = min(raw_a.n_times, raw_b.n_times)
    duration = (min_samples - 1) / sfreq

    epochs_a = mne.Epochs(
        raw_a, events_a, tmin=0, tmax=duration, baseline=None,
        preload=True, reject_by_annotation=False, verbose=False,
    )
    epochs_b = mne.Epochs(
        raw_b, events_b, tmin=0, tmax=duration, baseline=None,
        preload=True, reject_by_annotation=False, verbose=False,
    )

    mne.rename_channels(epochs_a.info, {ch: f"{ch}{suffix_a}" for ch in epochs_a.ch_names})
    mne.rename_channels(epochs_b.info, {ch: f"{ch}{suffix_b}" for ch in epochs_b.ch_names})

    combined = epochs_a.add_channels([epochs_b])
    logger.info("ダイアドエポックを結合しました（長さ: %.2f 秒）。", duration)
    return combined, sfreq
