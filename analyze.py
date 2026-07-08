import os
import numpy as np
import pandas as pd
import mne
from utils.analysis import prepare_whole_task_epochs, compute_whole_time_plv

def main():
    preprocesseddata_dir = "data/preprocessed"
    results_dir = "data/results_psi"
    
    # 結果保存用のフォルダを作成
    os.makedirs(results_dir, exist_ok=True)
    
    total_subjects = 3
    pairs = [(1, 2), (1, 3), (2, 3)]
    
    # 5つの主要周波数帯域の定義
    freq_bands = {
        'delta': np.arange(1, 4),
        'theta': np.arange(4, 8),
        'alpha': np.arange(8, 13),
        'beta':  np.arange(13, 31),
        'gamma': np.arange(31, 46)
    }

    for pair in pairs:
        p1, p2 = pair
        pair_label = f"pair_{p1:02d}_{p2:02d}"
        print(f"\n--- {pair_label} ---")
        
        file_p1 = f"{preprocesseddata_dir}/sub{p1}_preprocessed_raw.fif"
        file_p2 = f"{preprocesseddata_dir}/sub{p2}_preprocessed_raw.fif"
        
        if not os.path.exists(file_p1) or not os.path.exists(file_p2):
            print(f" [WARNING] 前処理済みファイルが足りません。スキップします。")
            continue
            
        # データの読み込みと合体
        raw_p1 = mne.io.read_raw_fif(file_p1, preload=True, verbose=False)
        raw_p2 = mne.io.read_raw_fif(file_p2, preload=True, verbose=False)
        combined_epochs, sfreq = prepare_whole_task_epochs(raw_p1, raw_p2)
        
        # CSVの行ラベルを生成
        ch_names_original = raw_p1.ch_names
        pair_row_labels = [f"{ch}_p1 <-> {ch}_p2" for ch in ch_names_original]
        
        # このペアの全帯域の結果を格納するデータフレームの土台
        df_pair_results = pd.DataFrame(index=pair_row_labels)
        
        # 各周波数帯域でPSIを計算してデータフレームに追加
        for band_name, freqs in freq_bands.items():
            print(f" -> 計算中: {band_name}帯域")
            psi_band_averaged = compute_whole_time_plv(combined_epochs, sfreq, freqs)
            
            # 列名（例: 'alpha'）としてPSIの1次元配列（32電極ペア分）を代入
            df_pair_results[band_name] = psi_band_averaged
            
        # CSVファイルへの書き出し
        csv_save_path = os.path.join(results_dir, f"{pair_label}_psi.csv")
        df_pair_results.to_csv(csv_save_path, index_label="Electrode_Pair")
        print(f" -> CSVを保存しました: {csv_save_path}")
        
    print("\n[INFO] すべてのペアのPSI算出、およびCSVエクスポートが完了しました！")

if __name__ == "__main__":
    main()