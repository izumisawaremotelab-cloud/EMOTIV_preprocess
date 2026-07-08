import numpy as np
import mne
from mne_connectivity import spectral_connectivity_epochs

def prepare_whole_task_epochs(raw_p1, raw_p2):
    """
    切り出したタスク全体のすべての時間（全サンプル）を1つの巨大な時間窓（1エポック）として定義し、
    2人分のデータを結合する。
    """
    sfreq = raw_p1.info['sfreq']
    
    # データの最初（0秒）に1つだけダミーのイベントを配置して、全時間を1つのエポックにする
    # イベント形式: [サンプル位置, 0, イベントID]
    events_p1 = np.array([[0, 0, 1]])
    events_p2 = np.array([[0, 0, 1]])
    
    # データの全体の長さを取得（tmaxに指定）
    total_duration_p1 = raw_p1.times[-1]
    total_duration_p2 = raw_p2.times[-1]
    # 安全のため、2人のデータ長のうち短い方に合わせる
    task_duration = min(total_duration_p1, total_duration_p2)
    
    # 全時間をそのまま切り出す（1エポック化）
    epochs_p1 = mne.Epochs(raw_p1, events_p1, tmin=0, tmax=task_duration, baseline=None, preload=True, verbose=False)
    epochs_p2 = mne.Epochs(raw_p2, events_p2, tmin=0, tmax=task_duration, baseline=None, preload=True, verbose=False)
    
    # 2人のチャンネル名が重複しないよう、末尾に _p1, _p2 を付与してリネーム
    mne.rename_channels(epochs_p1.info, {ch: f"{ch}_p1" for ch in epochs_p1.ch_names})
    mne.rename_channels(epochs_p2.info, {ch: f"{ch}_p2" for ch in epochs_p2.ch_names})
    
    # 2人の電極を1つのデータ構造に合体
    combined_epochs = mne.add_channels_epochs([epochs_p1, epochs_p2])
    
    return combined_epochs, sfreq

def compute_whole_time_plv(combined_epochs, sfreq, freqs):
    """
    タスク全時間を通したMorletウェーブレット変換を行い、
    指定された周波数帯域で平均化したPLV（PSI）を算出する。
    """
    n_ch = len(combined_epochs.ch_names) // 2
    indices = (np.arange(n_ch), np.arange(n_ch) + n_ch)

    con = spectral_connectivity_epochs(
        combined_epochs,
        method='plv',
        mode='cwt_morlet',
        cwt_freqs=freqs,       # 解析したい周波数帯域
        cwt_n_cycles=7.0,
        indices=indices,
        sfreq=sfreq,
        verbose=False
    )
    
    # 形状 (Shape): (ペア数, 周波数数)
    plv_raw_outputs = con.get_data()
    
    # 周波数軸（第1次元、引数 axis=1）に沿って平均をとる
    plv_band_averaged = np.mean(plv_raw_outputs, axis=1)
    
    return plv_band_averaged