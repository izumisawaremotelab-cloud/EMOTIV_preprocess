from eeg_preprocessing import crop_raw, apply_filters, run_ica

# 切り出し
raw_task, events = crop_raw('C:/Users/kouta/OneDrive/Desktop/Research/EMOTIV/515計測データ/F_Silent_FLEX2_659729_2026.05.15T18.17.08+09.00.md.bdf')

# フィルタ適用
raw_clean = apply_filters(raw_task)

raw_ica = run_ica(raw_clean)

raw_ica.save('subxx_preprocessed.fif', overwrite=True)