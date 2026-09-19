import numpy as np, pandas as pd

def fx_divergence_signal(domestic_hawk_score, foreign_hawk_score, fx_ofi_z=0.0, stance_weight=1.0, ofi_weight=0.5):
    return stance_weight*(domestic_hawk_score-foreign_hawk_score)+ofi_weight*fx_ofi_z

def credit_forward_regression(df: pd.DataFrame, ofi_col='single_name_ofi', index_spread_col='index_spread', horizon=3):
    d=df[[ofi_col,index_spread_col]].copy(); d['y']=d[index_spread_col].shift(-horizon)-d[index_spread_col]; d=d.dropna()
    x=d[ofi_col].to_numpy(float); y=d['y'].to_numpy(float); X=np.column_stack([np.ones(len(x)),x]); b,*_=np.linalg.lstsq(X,y,rcond=None); yh=X@b; sst=((y-y.mean())**2).sum(); sse=((y-yh)**2).sum()
    return {'alpha':float(b[0]),'beta':float(b[1]),'r2':float(1-sse/sst) if sst else np.nan,'n':len(d)}
