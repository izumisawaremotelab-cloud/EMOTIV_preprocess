"""
EEG前処理ユーティリティ（EMOTIV Flex 32ch対応）

主な機能:
  - mne_montage()   : デジタイザTXTからMNEモンタージュを生成
  - crop_raw()      : BDF読込 → マーカー区間で切り出し
  - apply_filters() : バンドパス + ノッチフィルタ
  - run_ica()       : ICA（仮想EOG + 筋電 + 尖度による自動検出付き）
  - get_segments()  : マーカーで条件分割（ICA前の分割用）
  - split_by_markers(): マーカーで条件分割して保存（ICA後の分割用）
"""

import os
import json
import warnings
from datetime import datetime

import mne
import numpy as np
from scipy import stats
from mne.preprocessing import ICA

# =====================================================================
# 定数定義
# =====================================================================

# EMOTIV補助チャンネルのプレフィックス（ICA対象から除外する）
AUX_CHANNEL_PREFIXES = (
    'CQ.', 'EQ.', 'MOT.',
    'TimestampS', 'TimestampMs', 'OrTimestamp',
    'Counter', 'Interpolated', 'RawCq', 'Battery', 'BatteryPercent',
    'FwBuffer', 'FwClock', 'MarkerHardware',
)

# 32チャネルのラベル（MATLAB順、デジタイザ点の並びに対応）
# mne_montage() と EXPECTED_EEG_CH_COUNT の双方から参照し、チャンネル数の
# 二重管理（ハードコードの重複）を避ける。
EEG_CHANNEL_LABELS = [
    'Cz', 'Fz', 'FP1', 'F7', 'F3', 'FC1', 'C3', 'FC5', 'FT9', 'T7',
    'CP5', 'CP1', 'P3', 'P7', 'PO9', 'O1', 'Pz', 'Oz', 'O2', 'PO10',
    'P8', 'P4', 'CP2', 'CP6', 'T8', 'FT10', 'FC6', 'C4', 'FC2', 'F4', 'F8', 'Fp2'
]

# 想定されるEEGチャンネル数（EMOTIV Flex 32ch）
EXPECTED_EEG_CH_COUNT = len(EEG_CHANNEL_LABELS)

# 注意: 条件分割定義はライブラリ側には置かない。
# 実験ごとに異なるため main.py 側で定義し、各関数に segments 引数として渡すこと。

# ===== ICA 解析条件パラメータ（変更が必要な場合はここを修正） ==========
# 論文や解析条件に合わせて調整するパラメータ群
ICA_VARIANCE_RATIO = 0.999999   # 分散の99.9999%を説明する成分数を自動選択
ICA_METHOD = 'infomax'          # ICA手法（'infomax', 'fastica', 'picard'）
ICA_RANDOM_STATE = 97           # 再現性のための乱数シード
MUSCLE_THRESHOLD = 0.8          # 筋電アーティファクト検出の閾値（高いほど緩い・低いほど厳しい）
                                 # 0.5=厳しい(過剰除外) / 0.8=標準(推奨) / 1.5=緩め
KURTOSIS_THRESHOLD = 20         # 突発的スパイク検出の尖度閾値（高いほど緩い）
                                 # 15=厳しい(過剰除外) / 20=標準(推奨) / 25=緩め

# 不良チャンネル自動検出の閾値（頑健Zスコア、MADベース）
# チャンネル間の振幅標準偏差のロバストZスコアがこれを超えると不良判定
BAD_CHANNEL_ZSCORE_THRESHOLD = 4.0
# =====================================================================


# =====================================================================
# モンタージュ生成
# =====================================================================

def mne_montage(sub_num, data_dir):
    """
    デジタイザTXTファイルからMNEカスタムモンタージュを生成する。

    Parameters
    ----------
    sub_num  : int  被験者番号
    data_dir : str  デジタイザファイルのディレクトリ

    Returns
    -------
    mne.channels.DigMontage
    """
    # デジタイザデータはsubxx.TXTの命名規則を想定
    input_filename = os.path.join(data_dir, f"sub{sub_num:02d}.TXT")

    if not os.path.exists(input_filename):
        raise FileNotFoundError(f"デジタイザファイルが見つかりません: {input_filename}")

    # デジタイザデータの全行読み込み
    raw_coords = []
    with open(input_filename, 'r', encoding='utf-8', errors='ignore') as f:
        for idx, line in enumerate(f):
            if idx >= 74:  # 一点につき2行のデータ、基準点5点 + 32ch
                break
            parts = line.split()
            if len(parts) >= 4:
                try:
                    # X, Y, Z 座標を抽出
                    raw_coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
                except ValueError:
                    continue

    coords_matrix = np.array(raw_coords)

    # --- 基準点（ランドマーク）とEEG電極の抽出 ---
    # デジタイザの生データ単位をMNE標準の「メートル」に変換するための除数
    # cm → m の場合: 100.0 / mm → m の場合: 1000.0
    DIGITIZER_UNIT_TO_METER = 100.0  # 現在のデータはcm単位

    # MATLAB: data(1:2:10, 1:3) -> 基準点
    # 0: Nz, 1: Iz, 2: RPA, 3: LPA, 4: Cz
    landmarks_m = coords_matrix[0:10:2] / DIGITIZER_UNIT_TO_METER

    # MATLAB: data(11:2:74, 1:3) -> 32個のEEG電極
    eeg_coords_m = coords_matrix[10:74:2] / DIGITIZER_UNIT_TO_METER

    # --- 辞書型の作成 ---
    # ランドマーク位置の対応付け
    fiducials = {
        'nasion': landmarks_m[0],   # Nz
        'rpa': landmarks_m[2],      # RPA
        'lpa': landmarks_m[3]       # LPA
    }

    # チャネルラベル（MATLAB順）はモジュール冒頭の EEG_CHANNEL_LABELS を使用
    ch_pos = {label: coord for label, coord in zip(EEG_CHANNEL_LABELS, eeg_coords_m)}

    # --- MNEカスタムモンタージュの生成 ---
    # デジタイザ空間（まだ標準脳に合わせる前）なので coord_frame='unknown' で開始
    montage = mne.channels.make_dig_montage(
        ch_pos=ch_pos,
        nasion=fiducials['nasion'],
        lpa=fiducials['lpa'],
        rpa=fiducials['rpa']
    )

    print("[INFO] 生デジタイザデータからMNEモンタージュを作成しました。")
    return montage


# =====================================================================
# データ読込・切り出し
# =====================================================================

def crop_raw(sub_num, data_dir, marker_start=2, marker_end=5):
    """
    BDFデータを読み込み、指定マーカー区間で切り出す。

    Parameters
    ----------
    sub_num      : int  被験者番号
    data_dir     : str  BDFファイルのディレクトリ
    marker_start : int  タスク開始マーカー値（デフォルト: 2）
    marker_end   : int  タスク終了マーカー値（デフォルト: 5）

    Returns
    -------
    (mne.io.Raw or None, ndarray)
        タスク区間のrawデータとイベント配列。マーカー未検出時はNoneを返す。
    """
    # データ読み込み、入力データはsubxx.bdfでの命名規則を想定
    input_filename = os.path.join(data_dir, f"sub{sub_num:02d}.bdf")
    raw = mne.io.read_raw_bdf(input_filename, preload=True)

    # マーカー値と時間情報の対応リスト
    events_list = []

    # サンプリング周波数の取得
    sfreq = raw.info['sfreq']

    # すべてのアノテーションに対してループ処理
    # ③ try-exceptでパース不能なアノテーションを安全にスキップ
    for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
        try:
            parts = desc.split(',')
            marker_value = int(parts[1])
        except (IndexError, ValueError):
            # Annotation形式が想定外の場合はスキップ
            continue
        sample_pos = int(np.round(onset * sfreq))
        events_list.append([sample_pos, 0, marker_value])

    # Numpy配列化（アノテーションが全くない場合のガード）
    if not events_list:
        print("\nedit error: アノテーションからマーカーを取得できませんでした")
        return None, np.empty((0, 3), dtype=int)
    events = np.array(events_list)
    print("loading events:\n", events)

    # 計測開始・終了のサンプルの取得
    start_samples = events[events[:, 2] == marker_start][:, 0]
    end_samples   = events[events[:, 2] == marker_end][:, 0]

    # マーカーが複数回検出された場合は、どちらを採用したかを明示する
    # （開始マーカーは最初の出現、終了マーカーは最後の出現を暗黙に採用しているため）
    if len(start_samples) > 1:
        print(f"[WARN] 開始マーカー({marker_start})が複数回検出されました: "
              f"{len(start_samples)}件。最初の出現を使用します。")
    if len(end_samples) > 1:
        print(f"[WARN] 終了マーカー({marker_end})が複数回検出されました: "
              f"{len(end_samples)}件。最後の出現を使用します。")

    if len(start_samples) > 0 and len(end_samples) > 0:
        start_time = start_samples[0] / sfreq
        end_time   = end_samples[-1] / sfreq
        raw_task = raw.copy().crop(tmin=start_time, tmax=end_time)
        print(f"\nedit success: {start_time:.2f} sec -- {end_time:.2f} sec")
        return raw_task, events
    else:
        print("\nedit error: No selected marker value")
        return None, events


# =====================================================================
# フィルタリング
# =====================================================================

def apply_filters(raw, bandpass_low=1.0, bandpass_high=60.0, notch_freqs=None):
    """
    バンドパスフィルタとノッチフィルタを適用する。

    Parameters
    ----------
    raw           : mne.io.Raw  対象データ
    bandpass_low  : float  バンドパス下限 (Hz)
    bandpass_high : float  バンドパス上限 (Hz)
    notch_freqs   : list   ノッチフィルタ周波数 (Hz)
    """
    if notch_freqs is None:
        notch_freqs = [50.0, 100.0]

    # バンドパスフィルタ
    raw_filtered = raw.copy().filter(
        l_freq=bandpass_low, h_freq=bandpass_high, fir_design='firwin'
    )

    # 電源ノイズ除去
    # MATLAB版(EEGLAB)のpop_cleanlineに合わせ、帯域を丸ごと削る帯域除去
    # フィルタではなく、マルチテーパー回帰で正弦波ノイズ成分だけを推定・
    # 除去する方式(spectrum_fit)を使う。
    raw_filtered.notch_filter(freqs=notch_freqs, method='spectrum_fit', verbose=False)

    return raw_filtered


# =====================================================================
# ICA（自動成分検出 + 手動確認）
# =====================================================================

def _extract_eeg_channels(raw):
    """EMOTIV補助チャンネルを除外し、EEGチャンネルのみを返す。"""
    eeg_chs = [ch for ch in raw.ch_names
               if not ch.startswith(AUX_CHANNEL_PREFIXES)]
    raw_eeg = raw.copy().pick(eeg_chs)
    return raw_eeg


def _detect_and_interpolate_bad_channels(raw_eeg, zscore_threshold=BAD_CHANNEL_ZSCORE_THRESHOLD):
    """
    振幅（標準偏差）のロバストZスコア（中央値・MAD基準）から不良チャンネルを
    自動検出し、モンタージュ位置を使って球面spline補間する（in-place）。

    フラット/過大振幅ないずれの外れチャンネルもICAの分離品質を下げるため、
    ICA fit前に処置する。

    Returns
    -------
    list[str]  不良と判定されたチャンネル名（補間済み）
    """
    data = raw_eeg.get_data(picks='eeg')
    stds = data.std(axis=1)

    median = np.median(stds)
    mad = stats.median_abs_deviation(stds, scale='normal')
    if mad == 0:
        return []

    robust_z = (stds - median) / mad
    bad_idx = np.where(np.abs(robust_z) > zscore_threshold)[0]
    bad_chs = [raw_eeg.ch_names[i] for i in bad_idx]

    if bad_chs:
        raw_eeg.info['bads'] = bad_chs
        raw_eeg.interpolate_bads(reset_bads=True, verbose=False)

    return bad_chs


def _save_ica_log(log_dir, sub_num, cond_name, ica):
    """
    ICA除外成分をJSONファイルに追記保存する（⑧ 再現性向上）。

    保存先: log_dir/ica_exclusion_log.json
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "ica_exclusion_log.json")

    # 既存ログの読み込み
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            log_data = json.load(f)
    else:
        log_data = {}

    # 新しいエントリの追加
    key = f"sub{sub_num:02d}_{cond_name}"
    log_data[key] = {
        "excluded": [int(i) for i in ica.exclude],
        "n_components": int(ica.n_components_),
        "method": ica.method,
        "timestamp": datetime.now().isoformat(),
    }

    # 書き込み
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)

    print(f"  [ログ保存] {log_path} ({key})")


def run_ica(raw, sub_num=None, cond_name=None, log_dir=None):
    """
    ICA（Independent Component Analysis）を実行する。

    仮想EOGチャンネル + 筋電 + 尖度 による3段階の自動成分検出を行い、
    検出された成分をグレー表示した状態で手動確認ウィンドウを表示する。

    Parameters
    ----------
    raw       : mne.io.Raw  対象データ
    sub_num   : int or None 被験者番号（ログ保存用、Noneなら保存しない）
    cond_name : str or None 条件名（ログ保存用）
    log_dir   : str or None ログ保存ディレクトリ

    Returns
    -------
    mne.io.Raw  ICA適用後のクリーンなデータ（EEGチャンネルのみ）
    """
    # --- EEGチャンネルのみ取り出す（EMOTIV補助chを除外） ---
    raw_eeg = _extract_eeg_channels(raw)
    n_eeg = len(raw_eeg.ch_names)
    print(f"  ICA対象チャンネル: {n_eeg} ch (EEGのみ)")

    # ⑥ EEG電極数チェック
    if n_eeg != EXPECTED_EEG_CH_COUNT:
        warnings.warn(
            f"EEGチャンネル数が想定と異なります: {n_eeg}ch "
            f"(想定: {EXPECTED_EEG_CH_COUNT}ch). "
            f"チャンネル: {raw_eeg.ch_names}"
        )

    # --- 不良チャンネルの自動検出・補間 ---
    # ノイズの多いチャンネル/フラットなチャンネルはICAの分離品質を下げるため、
    # ICA fit前（再参照前）に検出し、球面spline補間で置き換える。
    bad_chs = _detect_and_interpolate_bad_channels(raw_eeg)
    if bad_chs:
        print(f"  [自動検出] 不良チャンネル（補間済み）: {bad_chs}")
    else:
        print("  [自動検出] 不良チャンネルなし")

    # --- 平均参照（average reference） ---
    # ICAの分離安定性・後続解析の一貫性のため、EEGチャンネルのみを対象に
    # 平均参照を適用する。仮想EOGチャンネル(vEOG/hEOG)はこの後に作成する
    # bipolar差分であり、共通成分は差分計算で相殺されるため参照方式に
    # 依存しない。
    raw_eeg.set_eeg_reference('average', projection=False, verbose=False)

    # ============================================================
    # 自動ノイズ成分検出（論文の手法: 仮想EOGとの相関）
    # Dikker et al. (2017) Current Biology - Supplementary S1F
    # 垂直眼電(vEOG): Fp1 - F8（論文ではAF4 - F8）
    # 水平眼電(hEOG): F7 - F8
    # ============================================================

    # 仮想EOGチャンネルの作成
    mne.set_bipolar_reference(
        raw_eeg,
        anode=['Fp1', 'F7'],
        cathode=['F8', 'F8'],
        ch_name=['vEOG_proxy', 'hEOG_proxy'],
        drop_refs=False,
        copy=False
    )
    raw_eeg.set_channel_types({'vEOG_proxy': 'eog', 'hEOG_proxy': 'eog'})

    # ICA実行（EEGチャンネルのみフィット）
    # MATLAB版(EEGLAB)はフィルタ済みデータに直接ICAをかけているため、
    # それに合わせてICA用の追加ハイパスフィルタは挟まず、apply_filters()で
    # 既に1-60Hzバンドパス・電源ノイズ除去済みのraw_eegに直接fitする。
    # ⑦ 以下のパラメータは解析条件に応じて変更する（冒頭の定数を参照）
    ica = ICA(
        n_components=ICA_VARIANCE_RATIO,
        random_state=ICA_RANDOM_STATE,
        method=ICA_METHOD,
        fit_params=dict(extended=True)
    )
    ica.fit(raw_eeg, picks='eeg', verbose=False)
    print(f"  ICA成分数: {ica.n_components_}")

    # --- 自動検出 ---
    bad_auto = set()

    # 1. 垂直眼電（瞬き等）
    eog_idx_v, _ = ica.find_bads_eog(raw_eeg, ch_name='vEOG_proxy', verbose=False)
    if eog_idx_v:
        eog_idx_v = [int(i) for i in eog_idx_v]
        print(f"  [自動検出] 垂直眼電(瞬き等)と相関: {eog_idx_v}")
        bad_auto.update(eog_idx_v)

    # 2. 水平眼電（視線移動）
    eog_idx_h, _ = ica.find_bads_eog(raw_eeg, ch_name='hEOG_proxy', verbose=False)
    if eog_idx_h:
        eog_idx_h = [int(i) for i in eog_idx_h]
        print(f"  [自動検出] 水平眼電(視線移動)と相関: {eog_idx_h}")
        bad_auto.update(eog_idx_h)

    # 3. 筋肉ノイズ・体動（高周波数成分）
    # ⑦ threshold は解析条件に応じて変更（冒頭の MUSCLE_THRESHOLD を参照）
    muscle_idx, _ = ica.find_bads_muscle(raw_eeg, threshold=MUSCLE_THRESHOLD)
    if muscle_idx:
        muscle_idx = [int(i) for i in muscle_idx]
        print(f"  [自動検出] 高周波筋電/体動ノイズ: {muscle_idx}")
        bad_auto.update(muscle_idx)

    # 4. 突発的スパイク（高尖度）
    # ⑦ threshold は解析条件に応じて変更（冒頭の KURTOSIS_THRESHOLD を参照）
    sources = ica.get_sources(raw_eeg).get_data()
    kurt = stats.kurtosis(sources, axis=1)
    bad_kurt = np.where(kurt > KURTOSIS_THRESHOLD)[0].tolist()
    if bad_kurt:
        bad_kurt = [int(i) for i in bad_kurt]
        print(f"  [自動検出] 突発的スパイク (kurtosis>{KURTOSIS_THRESHOLD}): {bad_kurt}")
        bad_auto.update(bad_kurt)

    ica.exclude = sorted(list(bad_auto))
    print(f"  → 自動除外候補 合計: {ica.exclude}")
    print("\n[INFO] 自動検出結果（グレー）を確認し、必要に応じてクリックで修正後、閉じてください。")

    # タイトルに被験者番号と条件名を表示
    if sub_num is not None and cond_name:
        window_title = f"sub{sub_num:02d} / {cond_name}"
    else:
        window_title = "ICA"

    # グレーになった状態でウィンドウを表示（手動で追加/削除可能）
    ica.plot_sources(raw_eeg, block=True, title=f"{window_title} — ICA成分確認")

    # 最終的に選択された成分を確認
    print(f"除外対象として選択された成分: {ica.exclude}")

    # ⑧ ICA除外成分をJSONに保存（引数が指定されている場合）
    if log_dir and sub_num is not None and cond_name:
        _save_ica_log(log_dir, sub_num, cond_name, ica)

    # ICA適用（EEGチャンネルのみ）
    raw_clean = raw_eeg.copy()
    ica.apply(raw_clean)

    # 解析に不要になった仮想EOGチャンネルを削除
    raw_clean.drop_channels(['vEOG_proxy', 'hEOG_proxy'])

    # 除去後の波形確認
    print("\n[INFO] ICA除去後の脳波波形（Rawデータ）を表示します。")
    raw_clean.plot(block=True, title=f"{window_title} — ICA除去後")

    return raw_clean


# =====================================================================
# 条件分割（ICA前 / ICA後）
# =====================================================================

def _iter_segments(raw, events, segments):
    """
    マーカー区間をraw内の時刻に変換してyieldする内部ヘルパー。
    get_segments() と split_by_markers() の共通ロジックを担う。

    Yields
    ------
    (cond_name, m_start, m_end, t_start, t_end, seg)
    """
    sfreq = raw.info['sfreq']

    for (m_start, m_end), cond_name in segments.items():
        s_start_arr = events[events[:, 2] == m_start][:, 0]
        s_end_arr   = events[events[:, 2] == m_end][:, 0]

        if len(s_start_arr) == 0 or len(s_end_arr) == 0:
            print(f"  [SKIP] marker {m_start}→{m_end}: マーカー未検出")
            continue

        # BDFサンプル(絶対位置) → raw内の相対時刻(秒) に変換。
        # raw.first_samp は crop_raw() でのクロップ開始位置（絶対サンプル位置）を
        # MNEが保持している値なので、これを使えばタスク開始マーカーを
        # segmentsの中身から推測する必要がなく、CONDITION_SEGMENTSと
        # MARKER_TASK_STARTの値がズレても正しく動作する。
        t_start = (s_start_arr[0] - raw.first_samp) / sfreq
        t_end   = (s_end_arr[0]   - raw.first_samp) / sfreq
        t_start = max(t_start, raw.times[0])
        t_end   = min(t_end,   raw.times[-1])

        if t_start >= t_end:
            print(f"  [SKIP] marker {m_start}→{m_end}: 有効区間なし")
            continue

        seg = raw.copy().crop(tmin=t_start, tmax=t_end)
        yield cond_name, m_start, m_end, t_start, t_end, seg


def get_segments(raw, events, segments):
    """
    フィルタ済みrawをイベントマーカーで条件ごとに切り出してdictで返す。
    ICAを各条件に個別にかけるために使用する（ICA_MODE="segment"用）。

    Parameters
    ----------
    raw      : mne.io.Raw  フィルタ後・クロップ済みのデータ
    events   : ndarray     crop_raw() が返す events（BDF空間のサンプル位置）
    segments : dict        {(start_marker, end_marker): '条件名'}  ※必須・main.pyで定義

    Returns
    -------
    dict : {'条件名': (raw_segment, m_start, m_end)}
    """
    result = {}
    for cond_name, m_start, m_end, t_start, t_end, seg in _iter_segments(raw, events, segments):
        print(f"  [{cond_name}] marker {m_start}→{m_end}: "
              f"{t_start:.1f}〜{t_end:.1f}秒 ({seg.times[-1]:.1f}秒間)")
        result[cond_name] = (seg, m_start, m_end)
    return result


def save_segment(seg, out_dir, sub_num, cond_name, m_start, m_end):
    """
    条件別に切り出した/ICA処理済みのraw区間を保存する。
    split_by_markers()（ICA_MODE="whole"）と main.py の segmentモード
    （ICA_MODE="segment"）の両方から使う共通の保存処理。

    ファイル名: sub{sub_num}_{cond_name}_m{m_start}-m{m_end}_raw.fif

    Returns
    -------
    str  保存したファイルの絶対/相対パス
    """
    os.makedirs(out_dir, exist_ok=True)
    save_name = f"sub{sub_num}_{cond_name}_m{m_start}-m{m_end}_raw.fif"
    save_path = os.path.join(out_dir, save_name)
    seg.save(save_path, overwrite=True)
    return save_path


def split_by_markers(raw, events, out_dir, sub_num, segments):
    """
    ICA後のデータをイベントマーカーで条件ごとに分割して保存する。
    ICA_MODE="whole" で使用する。

    Parameters
    ----------
    raw      : mne.io.Raw  crop_raw() でクロップ済みの前処理後データ
    events   : ndarray     crop_raw() が返す events（BDF空間のサンプル位置）
    out_dir  : str         保存先ディレクトリ
    sub_num  : int         被験者番号
    segments : dict        {(start_marker, end_marker): '条件名'}  ※必須・main.pyで定義
    """
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[分割保存] sub{sub_num:02d}")
    for cond_name, m_start, m_end, t_start, t_end, seg in _iter_segments(raw, events, segments):
        save_path = save_segment(seg, out_dir, sub_num, cond_name, m_start, m_end)
        print(f"  [{cond_name}] {os.path.basename(save_path)}  "
              f"({t_start:.1f}〜{t_end:.1f}秒, {seg.times[-1]:.1f}秒間)")