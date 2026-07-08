import os
import numpy as np
import mne

# 各自作モジュールから関数をインポート
from utils.preprocessing import crop_raw, apply_filters, run_ica, mne_montage
from utils.analysis import prepare_whole_task_epochs, compute_whole_time_plv

def main():
    #脳波データ、デジタイザデータの保存先を指定
    eegdata_dir="data/raw/silent"
    digitizerdata_dir="data/digitizer"
    preprocesseddata_dir="data/preprocessed"
    subnum=3

    # 被験者数分データの切りだしとフィルタリング、モンタージュ情報の適用、ICAによるノイズ除去を行う
    subnumber=3
    for sub_num in range (1, subnumber+1):
        # 切り出し
        raw_task, events = crop_raw(sub_num, eegdata_dir)

        # モンタージュ適用
        montage = mne_montage(subnum, digitizerdata_dir)
        raw_task.set_montage(montage, match_case=False, on_missing='warn')

        # フィルタ適用
        raw_clean = apply_filters(raw_task)

        #ICAによるノイズ除去
        raw_ica = run_ica(raw_clean)

        #ファイル保存
        save_path = os.path.join(preprocesseddata_dir, f'sub{sub_num}_preprocessed_raw.fif')
        raw_ica.save(save_path, overwrite=True)

    
if __name__ == "__main__":
    main()