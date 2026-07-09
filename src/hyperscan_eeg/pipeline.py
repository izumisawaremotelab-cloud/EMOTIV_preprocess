"""前処理・同期指標算出の一連の流れをまとめた高水準オーケストレーション関数。

各関数は `config.py` のデータクラスをそのまま引数として受け取り、
被験者・ペア・条件・区間ラベルの組み合わせだけを外側のスクリプトが
制御する構成とする。これにより実行スクリプト側でのパラメータ変更・
実行制御が容易になる。
"""

from __future__ import annotations

import logging
from pathlib import Path

import mne
import pandas as pd

from . import connectivity, io, phase, preprocessing, visualization
from .config import (
    FilterConfig,
    FrequencyBand,
    ICAConfig,
    MarkerConfig,
    MontageConfig,
    PathConfig,
    SegmentPlan,
)
from .presets import EMOTIV_AUX_CHANNEL_PREFIXES

logger = logging.getLogger(__name__)


def run_subject_preprocessing(
    subject: int,
    condition: str,
    paths: PathConfig,
    marker_cfg: MarkerConfig,
    montage_cfg: MontageConfig,
    filter_cfg: FilterConfig = FilterConfig(),
    ica_cfg: ICAConfig = ICAConfig(),
    aux_prefixes: tuple[str, ...] = EMOTIV_AUX_CHANNEL_PREFIXES,
) -> Path:
    """1被験者・1条件分の生EEGデータを前処理し、fifとして保存する。

    切り出し -> モンタージュ適用 -> EEGチャンネル抽出 -> フィルタ -> ICA除去、
    の順で処理する。ゲーム状態（zz）区間へのさらなる分割は解析時
    （`run_pair_connectivity`）に行うため、ここではタスク全体を対象に
    ICAを1回だけフィットする（短い区間ごとに分割してからICAを行うと
    独立成分推定が不安定になりやすいため）。

    Returns:
        保存された前処理済みfifファイルのパス。
    """
    logger.info("=== 前処理開始: subject%d / %s ===", subject, condition)

    raw = io.load_raw_eeg(paths.raw_file(subject, condition))
    events = io.get_marker_events(raw, marker_cfg)
    raw_task = preprocessing.crop_to_task_window(raw, events, marker_cfg)

    montage = io.build_dig_montage(subject, paths.digitizer_dir, montage_cfg)
    raw_task.set_montage(montage, match_case=False, on_missing="warn")

    raw_eeg = preprocessing.select_eeg_channels(raw_task, aux_prefixes)
    raw_filtered = preprocessing.apply_bandpass_notch(raw_eeg, filter_cfg)

    ica = preprocessing.fit_ica(raw_filtered, ica_cfg)
    if ica_cfg.interactive:
        preprocessing.review_ica_interactively(ica, raw_filtered)
    raw_clean = preprocessing.apply_ica(raw_filtered, ica)

    save_path = paths.preprocessed_file(subject, condition)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    raw_clean.save(save_path, overwrite=True)
    logger.info("前処理済みデータを保存しました: %s", save_path)
    return save_path


def _load_segments(
    path: Path,
    plan: SegmentPlan | None,
) -> dict[str, mne.io.BaseRaw]:
    raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
    if plan is None:
        return {"whole_task": raw}
    events = io.get_marker_events(raw)
    return preprocessing.segment_by_markers(raw, events, plan)


def run_pair_connectivity(
    pair: tuple[int, int],
    condition: str,
    paths: PathConfig,
    bands: tuple[FrequencyBand, ...],
    segment_plan: SegmentPlan | None = None,
    method: str = "plv",
    save_csv: bool = True,
    save_figures: bool = False,
) -> dict[str, pd.DataFrame]:
    """1ペア・1条件分の脳間同期指標を、区間ごと・周波数帯域ごとに算出する。

    Args:
        segment_plan: ゲーム状態ごとの区間分割定義（例:
            `design.condition_segments[condition]`）。None の場合は
            タスク全体を単一区間 'whole_task' として扱う。
        method: `mne_connectivity.spectral_connectivity_epochs` の method。

    Returns:
        区間ラベルをキーとした {区間名: (電極ペア x 周波数帯域) DataFrame}。
    """
    sub_a, sub_b = pair
    pair_label = f"pair_{sub_a:02d}_{sub_b:02d}"
    logger.info("=== 同期指標算出開始: %s / %s ===", pair_label, condition)

    file_a = paths.preprocessed_file(sub_a, condition)
    file_b = paths.preprocessed_file(sub_b, condition)
    if not file_a.exists() or not file_b.exists():
        raise FileNotFoundError(f"前処理済みファイルが見つかりません: {file_a} または {file_b}")

    segments_a = _load_segments(file_a, segment_plan)
    segments_b = _load_segments(file_b, segment_plan)
    if segments_a.keys() != segments_b.keys():
        raise ValueError(
            f"2者間で検出された区間ラベルが一致しません: {sorted(segments_a)} vs {sorted(segments_b)}"
        )

    results: dict[str, pd.DataFrame] = {}
    for label in segments_a:
        logger.info("-> 区間: %s", label)
        combined_epochs, sfreq = phase.combine_dyad_epochs(segments_a[label], segments_b[label])
        df = connectivity.compute_all_bands(combined_epochs, sfreq, bands, method=method)
        results[label] = df

        if save_csv:
            csv_path = paths.results_dir / condition / label / f"{pair_label}_{method}.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index_label="Electrode_Pair")
            logger.info("CSVを保存しました: %s", csv_path)

        if save_figures:
            fig_path = paths.figures_dir / condition / label / f"{pair_label}_{method}.png"
            visualization.plot_band_heatmap(df, f"{pair_label} / {condition} / {label}", fig_path)

    return results
