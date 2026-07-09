"""2者間ダイアド解析のための位相計算前処理（チャンネル結合・疑似エポック化）。

`prepare_whole_task_epochs`（旧実装）は「タスク全体」専用だったが、
ここでは任意の2つのRaw区間（タスク全体でも、ゲーム状態ごとの
サブ区間でも）を受け取れる汎用関数として実装する。

補足: このチャンネル結合はPSI固有の要件ではない。
`mne_connectivity.spectral_connectivity_epochs` はPLV/PSI/coherence等
どの指標を使う場合でも「1つのEpochs/Rawオブジェクト内のチャンネル同士」の
接続性しか計算できないAPI仕様であるため、2者を比較するには両者の
チャンネルを1つのオブジェクトへ合体させる必要がある。

`combine_dyad_raw` は連続 `Raw` を返すところで止め、複数試行への
分割（疑似エポック化、`make_pseudo_trial_epochs`）は行わない。これは
PLV等のエポック平均を前提とする指標は1エポックだけでは常に自明な値
（PLVなら1）になってしまうため、`connectivity.compute_band_connectivity`
側で帯域ごとに適切な疑似エポック長を決めてから改めてエポック化する
必要があるからである（詳細は `connectivity.py` 参照）。
"""

from __future__ import annotations

import logging

import mne

logger = logging.getLogger(__name__)


def combine_dyad_raw(
    raw_a: mne.io.BaseRaw,
    raw_b: mne.io.BaseRaw,
    suffix_a: str = "_p1",
    suffix_b: str = "_p2",
) -> tuple[mne.io.BaseRaw, float]:
    """2者分のRaw区間を、チャンネル名衝突を避けつつ1つのRawに結合する。

    双方のチャンネル名が衝突しないよう `suffix_a` / `suffix_b` を付与した上で
    1つのRawオブジェクトにまとめる。区間長が異なる場合は短い方に合わせる。

    Raises:
        ValueError: サンプリング周波数が一致しない場合。
    """
    sfreq_a = raw_a.info["sfreq"]
    sfreq_b = raw_b.info["sfreq"]
    if sfreq_a != sfreq_b:
        raise ValueError(f"サンプリング周波数が一致しません: {sfreq_a} Hz vs {sfreq_b} Hz")
    sfreq = sfreq_a

    min_samples = min(raw_a.n_times, raw_b.n_times)
    duration = (min_samples - 1) / sfreq

    raw_a_cropped = raw_a.copy().crop(tmin=0, tmax=duration)
    raw_b_cropped = raw_b.copy().crop(tmin=0, tmax=duration)

    raw_a_cropped.rename_channels({ch: f"{ch}{suffix_a}" for ch in raw_a_cropped.ch_names})
    raw_b_cropped.rename_channels({ch: f"{ch}{suffix_b}" for ch in raw_b_cropped.ch_names})

    combined = raw_a_cropped.add_channels([raw_b_cropped])
    logger.info("ダイアドRawを結合しました（長さ: %.2f 秒）。", duration)
    return combined, sfreq


def make_pseudo_trial_epochs(raw: mne.io.BaseRaw, epoch_length: float) -> mne.Epochs:
    """連続Rawを固定長・非重複の疑似トライアルに分割する。

    `spectral_connectivity_epochs` のPLV等の指標はエポック間平均を前提と
    しており、1エポックのみでは自明な値になってしまう（詳細は
    `connectivity.py` 参照）。連続データにも同指標を適用できるよう、
    固定長ウィンドウを「複数トライアル」とみなして分割する。

    オーバーラップは付けない（オーバーラップさせるとウィンドウ間の
    独立性が失われ、指標が実際より高く出るバイアスにつながるため）。
    """
    return mne.make_fixed_length_epochs(
        raw, duration=epoch_length, preload=True,
        reject_by_annotation=False, overlap=0.0, verbose=False,
    )
