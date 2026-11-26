"""D42: fail-closed data audit and a small reproducible pose evaluator."""
from pathlib import Path
import argparse,copy,hashlib,json,platform,time
from collections import Counter,defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def audit(records):
    issues=[];seen=set();scenes=defaultdict(set);previous={}
    for r in records:
        rid=r['id']; key=(r['scene'],r['frame'],r['object'])
        def flag(code): issues.append(dict(id=rid,code=code))
        if key in seen: flag('duplicate_target')
        seen.add(key);scenes[r['scene']].add(r['split'])
        if r['unit']!='m':flag('unit_contract')
        R=np.asarray(r['R'],float);K=np.asarray(r['K'],float);t=np.asarray(r['t'],float)
        if R.shape!=(3,3) or not np.all(np.isfinite(R)) or not np.allclose(R.T@R,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(R),1,atol=1e-6):flag('invalid_rotation')
        if K.shape!=(3,3) or not np.isfinite(K).all() or K[0,0]<=0 or K[1,1]<=0:flag('invalid_intrinsics')
        if t.shape!=(3,) or not np.isfinite(t).all():flag('invalid_translation')
        if r['pred_t'] is not None:
            pred=np.asarray(r['pred_t'],float)
            if pred.shape!=(3,) or not np.isfinite(pred).all():flag('invalid_prediction')
        if not np.isfinite(r['stamp']):flag('invalid_timestamp')
        if r['depth_scale_m_per_unit']<=0:flag('depth_scale')
        if not r['calibration_id']:flag('missing_calibration')
        order=(r['split'],r['scene'])
        if order in previous and r['stamp']<=previous[order]:flag('nonmonotonic_time')
        previous[order]=r['stamp']
    for scene,splits in scenes.items():
        if len(splits)>1:issues.append(dict(id=f'scene:{scene}',code='scene_leakage'))
    return issues


def evaluate(records,threshold_m=.01):
    issues=audit(records)
    if issues:raise ValueError(f'audit failed: {len(issues)} issues; no official score')
    targets=[r for r in records if r['split']=='test']
    if not targets:raise ValueError('empty test target set')
    rows=[]
    for r in targets:
        # Fixed eight-vertex box, identity rotation; this is exactly ADD for pure translation.
        x=np.array([[a,b,c] for a in [-.03,.03] for b in [-.02,.02] for c in [-.05,.05]])
        gt=x+np.array(r['t'])
        error=None if r['pred_t'] is None else float(np.linalg.norm(x+np.array(r['pred_t'])-gt,axis=1).mean())
        rows.append(dict(id=r['id'],scene=r['scene'],error_m=error,present=error is not None,
                         success=error is not None and error<threshold_m))
    return dict(targets=len(rows),predictions=sum(r['present'] for r in rows),
                successes=sum(r['success'] for r in rows),recall=sum(r['success'] for r in rows)/len(rows),rows=rows)


def synthetic_records():
    rows=[]
    for i in range(60):
        scene=i//10; frame=i%10
        t=[.001*frame,0,1.]
        pred=None if i%4==0 else [t[0]+(.02 if i%3==0 else .003),0,1.]
        rows.append(dict(id=i,scene=scene,frame=frame,object=1,split=['train','val','test'][scene//2],
                         stamp=frame*.05,unit='m',R=np.eye(3).tolist(),t=t,pred_t=pred,
                         K=[[600,0,320],[0,600,240],[0,0,1]],depth_scale_m_per_unit=.001,
                         calibration_id='synthetic-pinhole-v1'))
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--threshold-mm',type=float,default=10)
    ap.add_argument('--out-dir',type=Path,default=Path(__file__).resolve().parents[2]/'_assets/3d_learning/day42')
    args=ap.parse_args()
    if args.threshold_mm<=0:raise ValueError('threshold must be positive')
    out=args.out_dir;out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    clean=synthetic_records();bad=copy.deepcopy(clean)
    bad[0]['unit']='mm';bad[12]['R'][0][0]=-1;bad[23]['K'][0][0]=None
    bad[31]['depth_scale_m_per_unit']=0;bad[44]['stamp']=.01;bad[54]['calibration_id']=''
    bad.append(copy.deepcopy(bad[59]))
    leak=copy.deepcopy(clean[0]);leak.update(id=999,frame=999,stamp=1.,split='test');bad.append(leak)
    issues=audit(bad)
    blocked=False
    try:evaluate(bad,args.threshold_mm/1000)
    except ValueError:blocked=True
    assert blocked and not audit(clean)
    a=evaluate(clean,args.threshold_mm/1000);b=evaluate(copy.deepcopy(clean),args.threshold_mm/1000)
    assert digest(a)==digest(b)
    counts=dict(Counter(q['code'] for q in issues))
    config=dict(seed=42,metric='ADD for identity-rotation translation-only box',threshold_m=args.threshold_mm/1000,
                comparator='<',missing_predictions='failure',split='test',target_policy='all specified test instances')
    manifest=dict(data_sha256=digest(clean),config_sha256=digest(config),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  evaluation_sha256=digest(a),python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__)
    result=dict(config=config,manifest=manifest,injected_record_count=len(bad),clean_record_count=len(clean),
                audit_issue_count=len(issues),audit_codes=counts,invalid_dataset_blocked=blocked,
                repeated_evaluation_identical=True,evaluation={k:v for k,v in a.items() if k!='rows'},
                elapsed_seconds=time.perf_counter()-start,
                scope='Synthetic module verification; no trained NOCS/EPro-PnP baseline, official benchmark or user assessment')
    for name,data in [('results',result),('manifest',manifest),('audit_before',issues),('clean_records',clean),('per_target',a['rows'])]:
        (out/f'{name}.json').write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    axes[0].barh(list(counts),list(counts.values()),color='#da6d3e');axes[0].set(xlabel='Issue count',title='Injected faults are not silently dropped')
    scenes=sorted(set(r['scene'] for r in a['rows']))
    y=[]
    for scene in scenes:
        subset=[r for r in a['rows'] if r['scene']==scene]
        y.append([sum(r['success'] for r in subset),sum(r['present'] and not r['success'] for r in subset),sum(not r['present'] for r in subset)])
    bottom=np.zeros(len(scenes))
    for j,(name,color) in enumerate([('success','#48a086'),('pose failure','#e5ae38'),('missing','#da6d3e')]):
        vals=np.array(y)[:,j];axes[1].bar([str(s) for s in scenes],vals,bottom=bottom,label=name,color=color);bottom+=vals
    axes[1].set(xlabel='Held-out scene',ylabel='Targets',title='Every target remains in the denominator');axes[1].legend()
    fig.savefig(out/'audit_report.png',dpi=160);plt.close(fig)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
