"""EEGデータ・デジタイザデータの読み込みとモンタージュ生成。"""

from __future__ import annotations

import logging
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from .config import MontageConfig

logger = logging.getLogger(__name__)


def load_raw_eeg(path: Path) -> mne.io.BaseRaw:
    """BDFファイルを読み込む。
    Raises:
        FileNotFoundError: 指定したパスにBDFファイルが見つからない場合。
    """

    if not path.exists():
        raise FileNotFoundError(f"EEGファイルが見つかりません: {path}")
    logger.info("EEGデータ->: %s", path)
    return mne.io.read_raw_bdf(path, preload=True, verbose=False)


def _marker_from_description(description: str, field_index: int) -> int | None:
    parts = description.split(",")
    if len(parts) <= field_index:
        return None
    try:
        return int(parts[field_index])
    except ValueError:
        return None


def _events_from_csv(raw: mne.io.BaseRaw, csv_file: Path) -> np.ndarray:
    """補正CSVを検証し、MNE events配列へ変換する。"""
    df = pd.read_csv(csv_file)
    required_columns = {"latency", "marker_value"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(
            f"マーカーCSVに必須列がありません: {csv_file} "
            f"(不足: {', '.join(sorted(missing))})"
        )
    if df.empty:
        raise ValueError(f"マーカーCSVが空です: {csv_file}")

    try:
        latencies = pd.to_numeric(df["latency"], errors="raise").to_numpy(dtype=float)
        marker_values = pd.to_numeric(
            df["marker_value"], errors="raise"
        ).to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"マーカーCSVのlatency/marker_valueに数値以外が含まれます: {csv_file}"
        ) from exc

    if not np.all(np.isfinite(latencies)) or not np.all(np.isfinite(marker_values)):
        raise ValueError(f"マーカーCSVに欠損値または無限値があります: {csv_file}")
    if np.any(latencies < 0):
        raise ValueError(f"マーカーCSVに負のlatencyがあります: {csv_file}")
    if not np.all(marker_values == np.round(marker_values)):
        raise ValueError(f"marker_valueは整数である必要があります: {csv_file}")
    if np.any(np.diff(latencies) < 0):
        raise ValueError(f"マーカーCSVのlatencyが時系列順ではありません: {csv_file}")

    samples = (
        np.round(latencies * raw.info["sfreq"]).astype(int) + raw.first_samp
    )
    last_sample = raw.first_samp + raw.n_times - 1
    if np.any(samples < raw.first_samp) or np.any(samples > last_sample):
        raise ValueError(
            f"マーカーCSVにBDF記録範囲外のイベントがあります: {csv_file}"
        )

    events = np.column_stack(
        [
            samples,
            np.zeros(len(df), dtype=int),
            marker_values.astype(int),
        ]
    )
    if len(np.unique(events[:, [0, 2]], axis=0)) != len(events):
        raise ValueError(
            f"マーカーCSVに同一サンプル・同一IDの重複があります: {csv_file}"
        )
    return events


def _replace_marker_annotations(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    field_index: int,
) -> None:
    """CSV補正イベントをRawへ反映し、FIF保存後にも引き継げるようにする。"""
    keep = np.array(
        [
            _marker_from_description(str(description), field_index) is None
            for description in raw.annotations.description
        ],
        dtype=bool,
    )
    preserved = raw.annotations[keep]
    event_desc = {
        int(marker): f"InletPort,{int(marker)},-1,csv"
        for marker in np.unique(events[:, 2])
    }
    corrected = mne.annotations_from_events(
        events,
        sfreq=raw.info["sfreq"],
        event_desc=event_desc,
        first_samp=raw.first_samp,
        orig_time=raw.info["meas_date"],
    )
    raw.set_annotations(preserved + corrected)


def get_marker_events(
    raw: mne.io.BaseRaw,
    csv_file: Path | None = None,
    allowed_markers: set[int] | None = None,
    field_index: int = 1,
) -> np.ndarray:
    """CSV補正またはBDF AnnotationsからMNE events配列を取得する。

    アノテーションの記述形式はカンマ区切りを想定し、``field_index`` はマーカー番号のフィールド位置
    （0始まり）を指定する。存在する補正CSVを優先し、補正イベントをRawの
    Annotationsにも反映するため、後段で保存したFIFから同じイベントを復元できる。
    ``allowed_markers`` を指定した場合、開始・終了・境界など解析に必要なIDだけを返す。

    Raises:
        ValueError: CSVが不正、または必要なマーカーが見つからない場合。
    """
    if csv_file is not None and csv_file.exists():
        events = _events_from_csv(raw, csv_file)
        _replace_marker_annotations(raw, events, field_index)
        logger.info("補正CSVのイベントを使用します: %s", csv_file)
    else:
        events, _ = mne.events_from_annotations(
            raw,
            event_id=lambda description: _marker_from_description(
                description, field_index
            ),
            verbose=False,
        )
        logger.info("BDF内Annotationsのイベントを使用します。")

    if allowed_markers is not None:
        events = events[np.isin(events[:, 2], list(allowed_markers))]
    if events.size == 0:
        raise ValueError(
            "解析対象のマーカーを含むイベントが見つかりません。"
            f" allowed_markers={allowed_markers}"
        )
    return events


def build_dig_montage(
    subject: int,
    digitizer_dir: Path,
    montage_cfg: MontageConfig,
) -> mne.channels.DigMontage:
    """デジタイザ生データ（sub<N>.TXT）からMNEカスタムモンタージュを生成する。

    ファイルは基準点5点 + 電極数の2倍行で構成される。
    行数は `n_landmarks` と `channel_labels` の数から動的に決定し、
    想定と食い違う場合は例外を送出する。

    生成直後の座標は digitizer 固有の任意座標系（coord_frame='unknown'）
    のままであり、このまま使うと頭部中心・向きが定まらない。旧MATLAB
    パイプラインではこの後 LORETA の
    「Utilities > Registering real electrodes > Arbitrary realistic
    coordinates」で鼻根(Nz)・両耳(LPA/RPA)を基準に頭部座標系へ登録して
    いたため、本関数末尾で同等の処理として
    `mne.channels.transform_to_head` を適用する。ただしこれは鼻根・
    両耳の3点を基準にした剛体変換（平行移動+回転）のみであり、
    電極間の相対距離（デジタイズ時の実測誤差等）そのものは補正しない
    点に注意（LORETAの当該機能も同様に剛体変換であると推定されるが、
    LORETA自体を実行できないため完全な検証はできていない）。

    Raises:
        FileNotFoundError: デジタイザファイルが存在しない場合。
        ValueError: 行数がチャンネル定義と整合しない場合。
    """
    input_path = digitizer_dir / f"sub{subject}.TXT"
    if not input_path.exists():
        raise FileNotFoundError(f"デジタイザファイルが見つかりません: {input_path}")

    n_channels = len(montage_cfg.channel_labels)
    n_points = montage_cfg.n_landmarks + n_channels
    n_lines_expected = n_points * 2

    coords: list[list[float]] = []
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError:
                continue
            if len(coords) >= n_lines_expected:
                break

    if len(coords) < n_lines_expected:
        raise ValueError(
            f"デジタイザファイルの座標行数が不足しています: "
            f"{input_path} (取得 {len(coords)} 行 / 必要 {n_lines_expected} 行)"
        )

    coords_matrix = np.array(coords)
    scale = montage_cfg.scale_to_meters

    # 各点2行記録のうち偶数行（1行目）を採用。基準点順: Nz, Iz, RPA, LPA, Cz
    landmarks_m = coords_matrix[0 : montage_cfg.n_landmarks * 2 : 2] / scale
    eeg_coords_m = coords_matrix[montage_cfg.n_landmarks * 2 : n_lines_expected : 2] / scale

    fiducials = {
        "nasion": landmarks_m[0],
        "rpa": landmarks_m[2],
        "lpa": landmarks_m[3],
    }
    ch_pos = dict(zip(montage_cfg.channel_labels, eeg_coords_m))

    montage = mne.channels.make_dig_montage(
        ch_pos=ch_pos,
        nasion=fiducials["nasion"],
        lpa=fiducials["lpa"],
        rpa=fiducials["rpa"],
    )
    montage = mne.channels.transform_to_head(montage)
    logger.info(
        "被験者%d: デジタイザデータからモンタージュを生成し、頭部座標系へ登録しました。", subject
    )
    return montage
