"""同期指標算出結果・モンタージュ配置の可視化。"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import pandas as pd

logger = logging.getLogger(__name__)


def plot_montage(
    montage: mne.channels.DigMontage,
    title: str | None = None,
    save_path: Path | None = None,
    show: bool = False,
) -> tuple[plt.Figure, plt.Figure]:
    """デジタイザから生成したモンタージュの電極配置を可視化し、目視で確認する。

    `build_dig_montage` が返す時点のモンタージュは coord_frame='unknown'
    のままであり、この状態で `montage.plot(kind='topomap')` を呼ぶと
    鼻根(nasion)・両耳(lpa/rpa)を基準にした頭部中心・向きが定まらず、
    電極が頭部モデルの中心付近に不自然に潰れて描画される
    （2D投影は頭部座標系を前提とするため）。そのため
    `mne.channels.transform_to_head` で正規化してから描画する。

    3D散布図・2Dトポマップの両方を返すので、電極のラベルが解剖学的に
    妥当な位置関係にあるか（前後・左右の対称性、Czが中心付近にあるか等）
    を目視で確認すること。既知の座標（例: T7-T8間、Fp1-O1間の距離が
    左右対称になっているか）を突き合わせて検証するとより確実。

    Args:
        show: True の場合、描画後に `plt.show()` でウィンドウ表示する
            （非対話バッチ実行では False のまま保存のみに留めること）。

    Returns:
        (3D散布図, 2Dトポマップ) のFigureタプル。
    """
    montage_head = mne.channels.transform_to_head(montage.copy())

    fig_3d = montage_head.plot(kind="3d", show_names=True, show=False)
    fig_topo = montage_head.plot(kind="topomap", show_names=True, show=False)
    if title:
        fig_3d.suptitle(f"{title} (3D)")
        fig_topo.suptitle(f"{title} (topomap)")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig_3d.savefig(save_path.with_stem(save_path.stem + "_3d"), dpi=150)
        fig_topo.savefig(save_path.with_stem(save_path.stem + "_topomap"), dpi=150)
        logger.info("モンタージュ確認図を保存しました: %s", save_path.parent)

    if show:
        plt.show()

    return fig_3d, fig_topo


def plot_ica_overlay(
    ica: mne.preprocessing.ICA,
    raw_before: mne.io.BaseRaw,
    save_path: Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """ICA除去前後の脳波を重ね描画し、除去成分の影響を確認する。

    `ica.exclude` に設定済みの成分を仮に除去した場合の波形を、
    除去前の波形に重ねて表示するMNE標準の確認図
    （`mne.preprocessing.ICA.plot_overlay`）を利用する。

    Args:
        raw_before: ICA適用前（フィルタ済み）のRaw。
        show: True の場合、描画後に `plt.show()` でウィンドウ表示する。

    Returns:
        除去前後の重ね描画Figure。
    """
    fig = ica.plot_overlay(raw_before, exclude=ica.exclude, show=False)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info("ICA除去前後の比較図を保存しました: %s", save_path)

    if show:
        plt.show()

    return fig


def plot_band_heatmap(
    matrix: pd.DataFrame,
    title: str,
    save_path: Path | None = None,
) -> plt.Figure:
    """1帯域分の p1側電極 x p2側電極 の同期指標をヒートマップとして描画する。"""
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=6)
    ax.set_xlabel(matrix.columns.name or "p2")
    ax.set_ylabel(matrix.index.name or "p1")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Connectivity")
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info("ヒートマップを保存しました: %s", save_path)
    return fig


def plot_band_means_bar(
    band_matrices: dict[str, pd.DataFrame],
    title: str,
    save_path: Path | None = None,
) -> plt.Figure:
    """各周波数帯域の（全電極ペア平均の）同期指標を棒グラフで描画する。"""
    band_names = list(band_matrices.keys())
    band_means = [matrix.to_numpy().mean() for matrix in band_matrices.values()]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(band_names, band_means)
    ax.set_ylabel("Mean connectivity")
    ax.set_title(title)
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info("棒グラフを保存しました: %s", save_path)
    return fig
