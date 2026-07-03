import mne
import numpy as np

def crop_raw(file_path):
    # データ読み込み
    raw = mne.io.read_raw_bdf(file_path, preload=True)

    # マーカー値と時間情報の対応リスト
    events_list = []

    # サンプリング周波数の取得
    sfreq = raw.info['sfreq']

    # すべてのアノテーションに対してループ処理
    for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
        parts = desc.split(',')
        marker_value = int(parts[1])
        sample_pos = int(np.round(onset * sfreq))
        events_list.append([sample_pos, 0, marker_value])

    # Numpy配列化
    events = np.array(events_list)
    print("loading events:\n", events)

    # 計測開始・終了のサンプルの取得、ここではstart:markervalue == 2, end:markervalue == 4
    start_samples = events[events[:, 2] == 2][:, 0]
    end_samples   = events[events[:, 2] == 4][:, 0]

    if len(start_samples) > 0 and len(end_samples) > 0:
        start_time = start_samples[0] / sfreq
        end_time   = end_samples[-1] / sfreq
        raw_task = raw.copy().crop(tmin=start_time, tmax=end_time)
        print(f"\nedit success: {start_time:.2f} sec -- {end_time:.2f} sec")
        return raw_task, events
    else:
        print("\nedit error: No selected marker value")
        return None, events

def apply_filters(raw):
    # バンドパス (1-60Hz)
    raw_filtered = raw.copy().filter(l_freq=1.0, h_freq=60.0, fir_design='firwin')
    
    # 電源ノイズ用ノッチ (50Hz & 100Hz)
    raw_filtered.notch_filter(freqs=[50.0, 100.0])
    
    return raw_filtered

def run_ica(raw):
    from mne.preprocessing import ICA
    
    # ハイパス (1Hz)　（推奨設定、バンドパスと重複してるが一応）
    raw_for_ica = raw.copy().filter(l_freq=1.0, h_freq=None, fir_design='firwin')
    
    # EEGLAB互換設定
    ica = ICA(n_components=None, random_state=97, method='infomax', fit_params=dict(extended=True))
    
    # 計算
    ica.fit(raw_for_ica)
    
    # 目視での除去
    # 画面を閉じると、選択した成分が自動で ica.exclude に格納される
    print("\n[INFO] 画面上で除外したいノイズ成分（瞬き・心拍等）を選択し、画面を閉じてください。")
    ica.plot_sources(raw, block=True)
    
    # 選択された成分を確認用に出力
    print(f"除外対象として選択された成分: {ica.exclude}")
    
    # 選択したノイズ成分を差し引く
    raw_clean = raw.copy()
    ica.apply(raw_clean)

    # --- 除去後の波形確認プロットを追加 ---
    print("\n[INFO] ICA除去後の脳波波形（Rawデータ）を表示します。")
    raw_clean.plot(block=True, title="ICA Cleaned Data")
    
    return raw_clean