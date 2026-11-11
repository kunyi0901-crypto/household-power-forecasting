"""小时单家庭预测：原始数据只读；目标时点为唯一键；仅验证集选模型。"""
from pathlib import Path
import json, zipfile, time, copy, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiment_outputs'; OUT.mkdir(exist_ok=True)
torch.set_num_threads(2)

def metrics(y,p):
    return {'MAE_kW':float(np.mean(abs(y-p))), 'RMSE_kW':float(np.sqrt(np.mean((y-p)**2)))}

def prepare():
    with zipfile.ZipFile(ROOT/'data/raw/household_power_consumption.zip') as z:
        with z.open('household_power_consumption.txt') as f:
            d=pd.read_csv(f,sep=';',usecols=['Date','Time','Global_active_power'],na_values=['?',''])
    idx=pd.DatetimeIndex(pd.to_datetime(d.Date+' '+d.Time,format='%d/%m/%Y %H:%M:%S'))
    assert idx.is_unique and idx.is_monotonic_increasing
    s=pd.Series(d.Global_active_power.to_numpy(),index=idx)
    h=s.resample('h').agg(['mean','count']); h.loc[h['count']<60,'mean']=np.nan
    assert h.index.equals(pd.date_range(h.index[0],h.index[-1],freq='h'))
    a,b=int(len(h)*.70),int(len(h)*.85)
    windows=np.lib.stride_tricks.sliding_window_view(h['mean'].to_numpy(),25)
    pos=np.arange(24,len(h)); valid=np.isfinite(windows).all(axis=1)
    audit=[]
    for name,mask in [('train',pos<a),('validation',(pos>=a)&(pos<b)),('test',pos>=b)]:
        audit.append(dict(split=name,candidate_windows=int(mask.sum()),valid_windows=int((mask&valid).sum()),excluded_missing=int((mask&~valid).sum())))
    pd.DataFrame(audit).to_csv(OUT/'sample_audit.csv',index=False)
    w=windows[valid]; pos=pos[valid]
    parts={name:(w[mask,:24].copy(),w[mask,24].copy(),h.index[pos[mask]]) for name,mask in [('train',pos<a),('validation',(pos>=a)&(pos<b)),('test',pos>=b)]}
    h.to_csv(OUT/'hourly_power.csv',index_label='hour_start')
    return parts

class GRU(nn.Module):
    def __init__(self):
        super().__init__(); self.rnn=nn.GRU(1,32,num_layers=2,batch_first=True); self.head=nn.Linear(32,1)
    def forward(self,x):
        out,_=self.rnn(x); return self.head(out[:,-1]).squeeze(-1)

def run():
    config=dict(lookback=24,horizon=1,seeds=[42,2024,2026],hidden=32,layers=2,batch_size=256,max_epochs=40,patience=6,learning_rate=.001,ridge_alphas=[.01,.1,1,10,100,1000],selection='validation MSE',plot_start='2010-05-01',plot_days=7)
    (OUT/'config.json').write_text(json.dumps(config,indent=2),encoding='utf8')
    parts=prepare(); x,y,_=parts['train']; vx,vy,_=parts['validation']
    # Ridge 每个滞后特征只按训练样本标准化。
    scaler=StandardScaler().fit(x); sx=scaler.transform(x); sv=scaler.transform(vx)
    candidates=[]; best=None; start=time.perf_counter()
    for alpha in config['ridge_alphas']:
        m=Ridge(alpha=alpha).fit(sx,y); score=metrics(vy,m.predict(sv))
        candidates.append(dict(alpha=alpha,**score))
        if best is None or score['RMSE_kW']<best[0]: best=(score['RMSE_kW'],m,alpha)
    ridge_seconds=time.perf_counter()-start
    pd.DataFrame(candidates).to_csv(OUT/'ridge_validation.csv',index=False)
    # GRU 输入与目标使用训练期同一功率尺度，验证测试不参与拟合。
    mu=float(x.mean()); sd=float(x.std())
    tx=torch.tensor((x-mu)/sd,dtype=torch.float32).unsqueeze(-1)
    ty=torch.tensor((y-mu)/sd,dtype=torch.float32)
    tv=torch.tensor((vx-mu)/sd,dtype=torch.float32).unsqueeze(-1)
    models=[]; logs=[]
    for seed in config['seeds']:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        model=GRU(); opt=torch.optim.Adam(model.parameters(),lr=.001)
        loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tx,ty),batch_size=256,shuffle=True)
        score_best=float('inf'); stale=0; start=time.perf_counter()
        for epoch in range(1,41):
            model.train(); total=0
            for bx,by in loader:
                opt.zero_grad(); loss=((model(bx)-by)**2).mean(); loss.backward(); opt.step(); total+=loss.item()*len(bx)
            model.eval()
            with torch.no_grad(): pred=model(tv).numpy()*sd+mu
            val=float(np.mean((pred-vy)**2)); logs.append(dict(seed=seed,epoch=epoch,train_MSE_kW2=total/len(tx)*sd**2,validation_MSE_kW2=val))
            if val<score_best:
                score_best=val; state=copy.deepcopy(model.state_dict()); best_epoch=epoch; stale=0
            else: stale+=1
            print(f'seed={seed} epoch={epoch} val_MSE={val:.5f}',flush=True)
            if stale>=6: break
        model.load_state_dict(state); model.eval(); elapsed=time.perf_counter()-start
        torch.save({'state_dict':state,'mean':mu,'std':sd,'best_epoch':best_epoch},OUT/f'gru_{seed}.pt')
        models.append((seed,model,best_epoch,elapsed))
    # 所有模型选择结束后才计算测试指标。
    rows=[]
    for split in ['validation','test']:
        px,py,dates=parts[split]; predictions=pd.DataFrame({'target_hour':dates,'actual_kW':py,'seasonal':px[:,0],'ridge':best[1].predict(scaler.transform(px))})
        for name in ['seasonal','ridge']:
            rows.append(dict(split=split,model=name,seed='',best_epoch='',fit_seconds=0 if name=='seasonal' else ridge_seconds,**metrics(py,predictions[name].to_numpy())))
        for seed,m,epoch,elapsed in models:
            with torch.no_grad(): p=m(torch.tensor((px-mu)/sd,dtype=torch.float32).unsqueeze(-1)).numpy()*sd+mu
            predictions[f'gru_{seed}']=p
            rows.append(dict(split=split,model='GRU',seed=seed,best_epoch=epoch,fit_seconds=elapsed,**metrics(py,p)))
        predictions.to_csv(OUT/f'{split}_predictions.csv',index=False)
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(logs).to_csv(OUT/'training_log.csv',index=False)
    (OUT/'selected_model.json').write_text(json.dumps(dict(ridge_alpha=best[2],gru_mean=mu,gru_std=sd,torch=torch.__version__,device='cpu'),indent=2))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
if __name__=='__main__': run()
