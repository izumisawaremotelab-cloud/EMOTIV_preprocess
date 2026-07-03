import mne
import numpy as np
Public: function 
#データ読み込み
raw = mne.io.read_raw_bdf('C:/Users/kouta/OneDrive/Desktop/Research/EMOTIV/515計測データ/F_Silent_FLEX2_659729_2026.05.15T18.17.08+09.00.md.bdf', preload=True)
#print(raw.annotations)

# マーカー値と時間情報の対応リスト
events_list = []

# サンプリング周波数の取得
sfreq = raw.info['sfreq']

# データに含まれるすべてのアノテーション（時間と文字列）をループ処理
for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
    
    # 例: 'InletPort,3,-1,2' をカンマで分割 -> ['InletPort', '3', '-1', '2']
    # parts[out/inlet, markevalue, duration, eventNumber]
    parts = desc.split(',')
    
    marker_value = int(parts[1])
           
    # 開始時間（秒）にサンプリング周波数を掛けて、サンプル番号を算出
    sample_pos = int(np.round(onset * sfreq))
            
    # MNE形式のイベントデータ [サンプル番号, 0, イベントID(markervalue)] を追加
    events_list.append([sample_pos, 0, marker_value])


# リストをNumPy配列に変換
events = np.array(events_list)

print("loading events:\n", events)

# 計測開始と計測終了のサンプルの取得、ここではstart:markervalue == 2, end:markervalue == 4
start_samples = events[events[:, 2] == 2][:, 0]
end_samples   = events[events[:, 2] == 4][:, 0]

if len(start_samples) > 0 and len(end_samples) > 0:
    start_time = start_samples[0] / sfreq
    end_time   = end_samples[-1] / sfreq
    
    raw_task = raw.copy().crop(tmin=start_time, tmax=end_time)
    print(f"\nedit success: {start_time:.2f} sec -- {end_time:.2f} sec")
else:
    print("\nedit error: No selected marker value")

