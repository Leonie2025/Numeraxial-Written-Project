from __future__ import annotations

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TICKERS = ["TY1", "ZN1", "RX1", "FF1", "SFR1"]


def infer_and_aggregate(path: Path) -> pd.DataFrame:
    use = ["timestamp","ticker","bid_price","ask_price","last_price","last_size",
           "bid_size","ask_size"]
    d = pd.read_csv(path, usecols=use, parse_dates=["timestamp"])
    at_ask = np.isclose(d["last_price"], d["ask_price"])
    at_bid = np.isclose(d["last_price"], d["bid_price"])
    sign = np.where(at_ask & ~at_bid, 1.0, np.where(at_bid & ~at_ask, -1.0, np.nan))
    miss = ~np.isfinite(sign)
    if miss.any():
        da = np.abs(d.loc[miss,"last_price"] - d.loc[miss,"ask_price"])
        db = np.abs(d.loc[miss,"last_price"] - d.loc[miss,"bid_price"])
        sign[miss] = np.where(da <= db, 1.0, -1.0)
    d["trade_sign"] = sign
    d["signed_volume"] = d["trade_sign"] * d["last_size"]
    d["mid"] = (d["bid_price"] + d["ask_price"]) / 2.0
    d = d.set_index("timestamp").sort_index()

    bars = d.resample("5min").agg(
        ticker=("ticker","first"),
        mid_open=("mid","first"), mid_close=("mid","last"),
        ofi=("signed_volume","sum"), volume=("last_size","sum"),
        n_trades=("last_size","size"), spread_mean=("ask_price", lambda x: np.nan),
    )
    bars["spread_mean"] = (d["ask_price"] - d["bid_price"]).resample("5min").mean()
    bars = bars.dropna(subset=["mid_close","ticker"])
    bars["session_date"] = bars.index.date
    bars["d_mid"] = bars.groupby("session_date")["mid_close"].diff()
    bars["ofi_ratio"] = bars["ofi"] / bars["volume"].replace(0,np.nan)
    return bars


def ols_lambda(df: pd.DataFrame):
    z = df[["ofi","d_mid"]].dropna()
    if len(z) < 5 or z["ofi"].var() == 0:
        return np.nan, np.nan, len(z)
    X = np.column_stack([np.ones(len(z)), z["ofi"].to_numpy(float)])
    y = z["d_mid"].to_numpy(float)
    b, *_ = np.linalg.lstsq(X,y,rcond=None)
    yh = X@b
    sst=((y-y.mean())**2).sum(); sse=((y-yh)**2).sum()
    return float(b[1]), float(1-sse/sst) if sst>0 else np.nan, len(z)


def rolling_ols(df: pd.DataFrame, window=24):
    x=df["ofi"]; y=df["d_mid"]
    cov=x.rolling(window,min_periods=max(8,window//2)).cov(y)
    var=x.rolling(window,min_periods=max(8,window//2)).var()
    return cov/var.replace(0,np.nan)


def hasbrouck_lambda_np(df: pd.DataFrame, lags=1, horizon=12):
    z=df[["ofi","d_mid"]].replace([np.inf,-np.inf],np.nan).dropna().copy()
    if len(z) < max(10, 4*(lags+1)):
        return np.nan
    sd=float(z["ofi"].std(ddof=1))
    if not np.isfinite(sd) or sd<=0: return np.nan
    arr=np.column_stack([(z["ofi"].to_numpy()-z["ofi"].mean())/sd, z["d_mid"].to_numpy()])
    T,k=arr.shape
    Y=arr[lags:]
    X=[np.ones(T-lags)]
    for L in range(1,lags+1):
        X.append(arr[lags-L:T-L])
    X=np.column_stack(X)
    try:
        B, *_=np.linalg.lstsq(X,Y,rcond=None)
        E=Y-X@B
        sigma=(E.T@E)/max(1,(len(E)-X.shape[1]))
        chol=np.linalg.cholesky(sigma + np.eye(2)*1e-12)
        As=[]
        for L in range(lags):
            As.append(B[1+2*L:1+2*(L+1),:].T)
        psi=[np.eye(2)]
        for h in range(1,horizon+1):
            ph=np.zeros((2,2))
            for i in range(1,min(lags,h)+1):
                ph += As[i-1] @ psi[h-i]
            psi.append(ph)
        cum=0.0
        for ph in psi:
            cum += float((ph@chol)[1,0])
        shock=float(chol[0,0]*sd)
        return cum/shock if abs(shock)>1e-15 else np.nan
    except np.linalg.LinAlgError:
        return np.nan


def full_day_hasbrouck(bars):
    vals=[]
    for _,g in bars.groupby("session_date"):
        v=hasbrouck_lambda_np(g,lags=1,horizon=12)
        if np.isfinite(v): vals.append(v)
    return (float(np.mean(vals)) if vals else np.nan,
            float(np.std(vals,ddof=1)) if len(vals)>1 else np.nan,
            len(vals))


def align_events(bars, events, ticker):
    rows=[]
    rols=rolling_ols(bars,24)
    for _,e in events.iterrows():
        t0=e["timestamp"]
        for rel in range(-30,16,5):
            t=t0+pd.Timedelta(minutes=rel)
            idx=bars.index.searchsorted(t,side="right")-1
            if idx<0: continue
            bt=bars.index[idx]
            if abs((bt-t).total_seconds())>301: continue
            day=bars.loc[bars["session_date"]==bt.date()]
            pos=day.index.get_indexer([bt])[0]
            lo=max(0,pos-11) 
            hb=hasbrouck_lambda_np(day.iloc[lo:pos+1],lags=1,horizon=8) if rel in (-30,-15,0,15) else np.nan
            rows.append({
                "ticker":ticker,"event_id":e["event_id"],"event":e["event"],
                "event_timestamp":t0,"relative_min":rel,
                "lambda_ols_rolling":float(rols.loc[bt]) if np.isfinite(rols.loc[bt]) else np.nan,
                "lambda_hasbrouck":hb,
                "ofi":float(bars.loc[bt,"ofi"]),
                "abs_ofi":abs(float(bars.loc[bt,"ofi"])),
                "surprise_std":float(e["surprise_std"]),
                "private_signal":float(e["private_signal"]),
            })
    return pd.DataFrame(rows)


def predictive_event_table(bars, events, ticker):
    rows=[]
    for _,e in events.iterrows():
        t0=e["timestamp"]
        pre=bars[(bars.index>=t0-pd.Timedelta(minutes=30)) & (bars.index<t0)]
        post=bars[(bars.index>=t0) & (bars.index<=t0+pd.Timedelta(minutes=15))]
        if len(pre)<2 or len(post)<1: continue
        pre_ofi=float(pre["ofi"].sum())
        p0=float(pre["mid_close"].iloc[-1])
        p1=float(post["mid_close"].iloc[-1])
        rows.append({
            "ticker":ticker,"event_id":e["event_id"],"event":e["event"],
            "timestamp":t0,"pre_ofi":pre_ofi,"announcement_move":p1-p0,
            "surprise_std":float(e["surprise_std"]),"private_signal":float(e["private_signal"]),
        })
    return pd.DataFrame(rows)


def predictive_stats(evdf):
    x=evdf["pre_ofi"].to_numpy(float); y=evdf["announcement_move"].to_numpy(float)
    if len(x)<5 or np.var(x)==0: return {}
    X=np.column_stack([np.ones(len(x)),x]); b,*_=np.linalg.lstsq(X,y,rcond=None)
    yh=X@b; sst=((y-y.mean())**2).sum(); sse=((y-yh)**2).sum()
    valid=(x!=0)&(y!=0)
    sign_acc=float(np.mean(np.sign(x[valid])==np.sign(y[valid]))) if valid.any() else np.nan
    return {"n_events":len(x),"slope_move_on_pre_ofi":float(b[1]),
            "r2":float(1-sse/sst) if sst>0 else np.nan,
            "corr":float(np.corrcoef(x,y)[0,1]),"sign_accuracy":sign_acc}


def plot_event(es,out):
    for metric,title,ylabel in [
        ("lambda_ols_rolling","Event-time rolling OLS lambda","Estimated lambda"),
        ("lambda_hasbrouck","Event-time Hasbrouck-style lambda (median across events)","Estimated lambda"),
        ("abs_ofi","Event-time absolute OFI","Mean |OFI|")]:
        fig,ax=plt.subplots(figsize=(8,4.5))
        for ticker,g in es.groupby("ticker"):
            grouped = g.groupby("relative_min")[metric]
            s = grouped.median() if metric == "lambda_hasbrouck" else grouped.mean()
            ax.plot(s.index,s.values,marker="o",ms=3,label=ticker)
        ax.axvline(0,ls="--",lw=1)
        ax.set_title(title); ax.set_xlabel("Minutes relative to release"); ax.set_ylabel(ylabel)
        ax.legend(ncol=3,fontsize=8); fig.tight_layout(); fig.savefig(out/f"{metric}.png",dpi=160); plt.close(fig)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data-dir",default="data_scenario2"); ap.add_argument("--out-dir",default="results")
    args=ap.parse_args(); data=Path(args.data_dir); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    figdir=out/"figures"; figdir.mkdir(exist_ok=True)
    events=pd.read_csv(data/"macro_events.csv",parse_dates=["timestamp"])

    summaries=[]; all_es=[]; all_pred=[]
    for ticker in TICKERS:
        print("Analyzing",ticker,flush=True)
        bars=infer_and_aggregate(data/f"{ticker}.csv.gz")
        lam,r2,n=ols_lambda(bars); hbmean,hbsd,hbn=full_day_hasbrouck(bars)
        true_lambda=float(pd.read_csv(data/f"{ticker}.csv.gz",usecols=["lambda_true"],nrows=1)["lambda_true"].iloc[0])
        summaries.append({"ticker":ticker,"true_lambda":true_lambda,"ols_lambda":lam,"ols_r2":r2,"n_5min_bars":n,
                          "hasbrouck_daily_mean":hbmean,"hasbrouck_daily_sd":hbsd,"hasbrouck_days":hbn})
        es=align_events(bars,events,ticker); all_es.append(es)
        pred=predictive_event_table(bars,events,ticker); all_pred.append(pred)
        bars[["mid_close","d_mid","ofi","ofi_ratio","volume","n_trades"]].to_csv(out/f"{ticker}_5min.csv.gz",compression="gzip")

    summary=pd.DataFrame(summaries); summary.to_csv(out/"lambda_summary.csv",index=False)
    es=pd.concat(all_es,ignore_index=True); es.to_csv(out/"event_time_metrics.csv",index=False)
    pred=pd.concat(all_pred,ignore_index=True); pred.to_csv(out/"preannouncement_prediction.csv",index=False)

    pstats=[]
    for ticker,g in pred.groupby("ticker"):
        z=predictive_stats(g); z["ticker"]=ticker; pstats.append(z)
    pd.DataFrame(pstats).to_csv(out/"prediction_summary.csv",index=False)
    rows=[]
    for ticker,g in es.groupby("ticker"):
        for metric in ["lambda_ols_rolling","lambda_hasbrouck","abs_ofi"]:
            aggregation = "median" if metric == "lambda_hasbrouck" else "mean"
            pre_values = g[g.relative_min==-30][metric]
            event_values = g[g.relative_min>=0][metric]
            if aggregation == "median":
                pre = pre_values.median()
                event = event_values.median()
            else:
                pre = pre_values.mean()
                event = event_values.mean()
            rows.append({
                "ticker":ticker,
                "metric":metric,
                "aggregation":aggregation,
                "pre_-30":pre,
                "event_0_15":event,
                "ratio_event_to_pre":event/pre if np.isfinite(pre) and abs(pre)>1e-15 else np.nan,
            })
    regime=pd.DataFrame(rows); regime.to_csv(out/"event_vs_pre_summary.csv",index=False)
    plot_event(es,figdir)

    payload={"n_macro_events":int(len(events)),"period":"2021-2025","scenario":"macro-informed flow; constant structural lambda",
             "lambda_summary":summary.to_dict(orient="records"),"prediction_summary":pstats}
    (out/"results.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(summary.to_string(index=False)); print(pd.DataFrame(pstats).to_string(index=False))

if __name__=="__main__": 
    main()