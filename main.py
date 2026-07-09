"""
EMOTIV Flex (32ch) EEG前処理メインスクリプト

処理フロー:
  1. BDF読込 → タスク区間（marker2〜marker5）切り出し
  2. デジタイザモンタージュ適用
  3. バンドパス + ノッチフィルタ
  4. ICA（自動成分検出 + 手動確認）
  5. 条件別に保存

ICA_MODEで処理方式を切り替え可能:
  "segment" : 条件ごとにICA → 保存（予備実験向け）
  "whole"   : タスク全体にICA → 条件分割して保存（本実験向け）
"""

import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

from utils.eeg_preprocessing import (
    crop_raw, apply_filters, run_ica, mne_montage,
    get_segments, split_by_markers, save_segment,
)

# =====================================================================
# 定数定義（⑨ マジックナンバーの定数化）
# =====================================================================

# --- パス設定 ---
EEGDATA_DIR        = "C:/Users/naohi/M2study/EMOTIV_preprocess-main/data/raw"
DIGITIZERDATA_DIR  = "C:/Users/naohi/M2study/EMOTIV_preprocess-main/data/digitizer"
PREPROCESSED_DIR   = "C:/Users/naohi/M2study/EMOTIV_preprocess-main/data/preprocessed"

# --- 被験者数 ---
SUBNUMBER = 1

# --- マーカー定義 ---
MARKER_TASK_START = 2   # タスク開始マーカー値
MARKER_TASK_END   = 5   # タスク終了マーカー値

# --- 条件分割定義 ---
# {(開始マーカー, 終了マーカー): '条件名'}
CONDITION_SEGMENTS = {
    (2, 3): 'script',   # 台本
    (3, 4): 'nod',      # うなずきあり
    (4, 5): 'nonod',    # うなずきなし
}

# --- フィルタ設定 ---
BANDPASS_LOW  = 1.0             # バンドパスフィルタ下限 (Hz)
BANDPASS_HIGH = 60.0            # バンドパスフィルタ上限 (Hz)
NOTCH_FREQS   = [50.0, 100.0]  # ノッチフィルタ周波数 (Hz)

# --- ICA処理モード（① 切替設定） ---
# "segment" : 条件ごとにICAを適用（予備実験向け）
#   - 各条件の固有アーティファクトを個別に除去できる
#   - うなずき条件特有のノイズが他条件のICA分離に影響しない
# "whole"   : タスク全体にICAを適用し、その後markerで分割（本実験向け）
#   - 全条件で一貫したICA成分分離が得られる
#   - 統計比較時に条件間のバイアスを避けられる
ICA_MODE = "segment"

# =====================================================================
# メイン処理
# =====================================================================

# ④ 保存先ディレクトリの自動作成
os.makedirs(PREPROCESSED_DIR, exist_ok=True)

for sub_num in range(1, SUBNUMBER + 1):
    print(f"\n========== sub{sub_num:02d} 前処理開始 ==========")

    # --- 1. データ読み込み・タスク区間の切り出し ---
    raw_task, events = crop_raw(
        sub_num, EEGDATA_DIR,
        marker_start=MARKER_TASK_START,
        marker_end=MARKER_TASK_END,
    )

    # ② crop_rawがNoneを返した場合はこの被験者をスキップ
    if raw_task is None:
        print(f"[SKIP] sub{sub_num:02d}: タスク区間が見つかりません")
        continue

    # --- 2. モンタージュ適用 ---
    montage = mne_montage(sub_num, DIGITIZERDATA_DIR)
    raw_task.set_montage(montage, match_case=False, on_missing='warn')

    # --- 3. フィルタ適用 ---
    raw_filtered = apply_filters(
        raw_task,
        bandpass_low=BANDPASS_LOW,
        bandpass_high=BANDPASS_HIGH,
        notch_freqs=NOTCH_FREQS,
    )

    # --- 4 & 5. ICA → 保存（モードで分岐） ---
    if ICA_MODE == "whole":
        # ===== タスク全体でICA → その後条件分割して保存 =====
        print(f"\n[ICA_MODE=whole] タスク全体にICA適用")
        raw_ica = run_ica(
            raw_filtered,
            sub_num=sub_num,
            cond_name="whole",
            log_dir=PREPROCESSED_DIR,
        )
        # ICA適用後のデータを条件分割して保存
        split_by_markers(
            raw_ica, events, PREPROCESSED_DIR, sub_num,
            segments=CONDITION_SEGMENTS,
        )

    elif ICA_MODE == "segment":
        # ===== 条件ごとにICA → 保存 =====
        print(f"\n[ICA_MODE=segment] 条件ごとにICA適用")
        print(f"\n[分割] sub{sub_num:02d}")
        seg_dict = get_segments(raw_filtered, events, segments=CONDITION_SEGMENTS)

        for cond_name, (seg, m_start, m_end) in seg_dict.items():
            print(f"\n--- {cond_name} (marker {m_start}→{m_end}) ICA開始 ---")

            seg_ica = run_ica(
                seg,
                sub_num=sub_num,
                cond_name=cond_name,
                log_dir=PREPROCESSED_DIR,
            )

            # 保存
            save_path = save_segment(seg_ica, PREPROCESSED_DIR, sub_num, cond_name, m_start, m_end)
            print(f"[保存] {save_path}")

    else:
        raise ValueError(f"不明なICA_MODE: '{ICA_MODE}'. 'whole' または 'segment' を指定してください。")

print("\n========== 全被験者の前処理が完了しました ==========")