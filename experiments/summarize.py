from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1]; O=P/'experiment_outputs'
m=pd.read_csv(O/'metrics.csv'); t=m[m.split=='test']; g=t[t.model=='GRU']; r=t[t.model=='ridge'].iloc[0]; b=t[t.model=='seasonal'].iloc[0]
a=pd.read_csv(O/'sample_audit.csv')
log=pd.read_csv(O/'training_log.csv')
fig,axes=plt.subplots(1,3,figsize=(12,3.5),constrained_layout=True)
for ax,(seed,d) in zip(axes,log.groupby('seed')):
 ax.plot(d.epoch,d.train_MSE_kW2,label='Train'); ax.plot(d.epoch,d.validation_MSE_kW2,label='Validation'); ax.set(title=f'Seed {seed}',xlabel='Epoch',ylabel='MSE (kW squared)'); ax.legend()
fig.savefig(O/'learning_curves.png',dpi=160); plt.close(fig)
p=pd.read_csv(O/'test_predictions.csv',parse_dates=['target_hour']).set_index('target_hour')
w=p.loc['2010-05-01':'2010-05-07'].asfreq('h')
fig,ax=plt.subplots(figsize=(12,4),constrained_layout=True)
for col in ['actual_kW','seasonal','ridge','gru_42']: ax.plot(w.index,w[col],label=col,linewidth=1,alpha=.8)
ax.set(ylabel='Power (kW)',xlabel='Target hour',title='Preselected test week; GRU seed fixed at 42'); ax.legend(ncol=4);fig.savefig(O/'forecast_week.png',dpi=160);plt.close(fig)
# 独立从逐时预测重新核算所有测试指标。
for _,row in t.iterrows():
 col=f'gru_{int(row.seed)}' if row.model=='GRU' else row.model
 err=p[col]-p.actual_kW
 assert np.isclose(abs(err).mean(),row.MAE_kW)
 assert np.isclose(np.sqrt((err**2).mean()),row.RMSE_kW)

print(t.to_string(index=False))
print('Plots saved; prediction metrics independently verified.')
