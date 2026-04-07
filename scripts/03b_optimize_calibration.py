#!/usr/bin/env python3
"""
Optimise calibration by trying sample subsets and outlier removal.
No robot/camera needed — works from saved samples.

Usage:
    python scripts/03b_optimize_calibration.py
    python scripts/03b_optimize_calibration.py --target-pos-mm 2.0
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cv2
from datetime import datetime
from scipy.spatial.transform import Rotation as R

METHODS = [
    (cv2.CALIB_HAND_EYE_TSAI,       "TSAI"),
    (cv2.CALIB_HAND_EYE_PARK,       "PARK"),
    (cv2.CALIB_HAND_EYE_HORAUD,     "HORAUD"),
    (cv2.CALIB_HAND_EYE_ANDREFF,    "ANDREFF"),
    (cv2.CALIB_HAND_EYE_DANIILIDIS, "DANIILIDIS"),
]


def calibrate(samples, mid):
    Rb, tb, Rc, tc = [], [], [], []
    for s in samples:
        B = np.array(s['T_base_tcp']); C = np.array(s['T_cam_board'])
        Rb.append(B[:3,:3]); tb.append(B[:3,3:4])
        Rc.append(C[:3,:3]); tc.append(C[:3,3:4])
    return cv2.calibrateHandEye(Rb, tb, Rc, tc, method=mid)


def error(Rtc, ttc, samples):
    X = np.eye(4); X[:3,:3]=Rtc; X[:3,3]=ttc.flatten(); Xi=np.linalg.inv(X)
    pe, re = [], []
    n = len(samples)
    for i in range(n):
        Ai = np.array(samples[i]['T_base_tcp'])
        Ci = np.array(samples[i]['T_cam_board'])
        for j in range(i+1, n):
            Aj = np.array(samples[j]['T_base_tcp'])
            Cj = np.array(samples[j]['T_cam_board'])
            A = np.linalg.inv(Ai)@Aj; B = Ci@np.linalg.inv(Cj)
            Ae = X@B@Xi
            pe.append(float(np.linalg.norm(A[:3,3]-Ae[:3,3])))
            Rd = A[:3,:3]@Ae[:3,:3].T
            tr = np.clip(np.trace(Rd),-1.,3.)
            re.append(float(np.arccos(np.clip((tr-1)/2,-1.,1.))))
    return {'pos_mm': float(np.median(pe))*1000 if pe else 999,
            'rot_deg': float(np.degrees(np.median(re))) if re else 999,
            'cons_mm': float(np.std(pe))*1000 if pe else 999}


def try_one(samples, mid, mname):
    if len(samples) < 10: return None
    try:
        Rtc, ttc = calibrate(samples, mid)
        e = error(Rtc, ttc, samples)
        T = np.eye(4); T[:3,:3]=Rtc; T[:3,3]=ttc.flatten()
        return dict(method=mname, T_tcp_cam=T, n=len(samples), **e)
    except Exception:
        return None


def strat_reproj(samples, mn):
    results = []
    for t in [0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.80]:
        sub = [s for s in samples if s['reproj_error_px'] <= t]
        if len(sub) < mn: continue
        for mid, mname in METHODS:
            r = try_one(sub, mid, mname)
            if r: r['filter']=f"reproj≤{t:.2f}px"; results.append(r)
    return results


def strat_corners(samples, mn):
    results = []
    for mc in [10,12,14,16,18,20,22,24]:
        sub = [s for s in samples if s['corner_count'] >= mc]
        if len(sub) < mn: continue
        for mid, mname in METHODS:
            r = try_one(sub, mid, mname)
            if r: r['filter']=f"corners≥{mc}"; results.append(r)
    return results


def strat_combined(samples, mn):
    results = []
    for rp in [0.35,0.40,0.45,0.50,0.55]:
        for mc in [10,12,14,16,18,20]:
            sub = [s for s in samples
                   if s['reproj_error_px']<=rp and s['corner_count']>=mc]
            if len(sub)<mn: continue
            for mid, mname in METHODS:
                r = try_one(sub, mid, mname)
                if r: r['filter']=f"reproj≤{rp:.2f}+corners≥{mc}"; results.append(r)
    return results


def strat_angle(samples, mn):
    results = []
    for lo in [10,15,20]:
        for hi in [40,45,50,55]:
            for rp in [0.45,0.55,0.80]:
                sub = [s for s in samples
                       if lo<=s['viewing_angle_deg']<=hi and s['reproj_error_px']<=rp]
                if len(sub)<mn: continue
                for mid, mname in METHODS:
                    r = try_one(sub, mid, mname)
                    if r: r['filter']=f"angle[{lo}-{hi}]+reproj≤{rp:.2f}"; results.append(r)
    return results


def strat_outlier(samples, mn):
    results = []
    for mid, mname in METHODS:
        cur = list(samples)
        base = try_one(cur, mid, mname)
        if not base: continue
        best = base; removed = []
        for _ in range(min(15, len(samples)-mn)):
            bi, bw, br = -999., -1, None
            for i in range(len(cur)):
                sub = cur[:i]+cur[i+1:]
                r = try_one(sub, mid, mname)
                if r:
                    imp = best['pos_mm']-r['pos_mm']
                    if imp > bi: bi,bw,br = imp,i,r
            if bw<0 or bi<0.005: break
            removed.append(cur[bw]['id']); cur.pop(bw); best = br
        if removed:
            best['filter']=f"outlier_rm({mname}): removed {','.join(removed)}"
            results.append(best)
    return results


def save_best(result, intr_file, tp, tr):
    T = result['T_tcp_cam']; Ti = np.linalg.inv(T)
    trans = T[:3,3]; quat = R.from_matrix(T[:3,:3]).as_quat()
    euler = R.from_matrix(T[:3,:3]).as_euler('xyz', degrees=True)
    hit = bool(result['pos_mm']<tp and result['rot_deg']<tr)
    try:
        with open(intr_file) as f: idata = json.load(f)
        intr = idata.get('intrinsics', idata)
        cm = [[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1]]
        dc = intr.get('distortion',[0]*5)
    except Exception:
        cm, dc = np.eye(3).tolist(), [0]*5

    data = {
        'metadata': {'timestamp': datetime.now().isoformat(),
                      'method': result['method'],
                      'num_samples': result['n'],
                      'filter_strategy': result['filter'],
                      'target_achieved': hit},
        'accuracy': {'position_error_mm': float(result['pos_mm']),
                      'rotation_error_deg': float(result['rot_deg']),
                      'consistency_mm': float(result['cons_mm'])},
        'transformations': {'T_tcp_cam': T.tolist(), 'T_cam_tcp': Ti.tolist(),
                             'translation_m': trans.tolist(),
                             'quaternion_xyzw': quat.tolist(),
                             'euler_xyz_deg': euler.tolist()},
        'camera_intrinsics': {
            'camera_matrix': cm if isinstance(cm,list) else cm.tolist(),
            'dist_coeffs': dc if isinstance(dc,list) else dc.tolist()}}
    os.makedirs('results/calibration', exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for fp in [f"results/calibration/optimized_{ts}.json",
               "results/calibration/calibration_latest.json"]:
        with open(fp,'w') as f: json.dump(data,f,indent=2)
        print(f"  💾 {fp}")
    return hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', default='results/samples/samples_latest.json')
    ap.add_argument('--target-pos-mm', type=float, default=2.0)
    ap.add_argument('--target-rot-deg', type=float, default=0.5)
    ap.add_argument('--min-samples', type=int, default=12)
    args = ap.parse_args()

    with open(args.samples) as f: data = json.load(f)
    samples = data['samples']

    rp = [s['reproj_error_px'] for s in samples]
    cn = [s['corner_count'] for s in samples]
    print(f"\n{'='*70}")
    print(f"📊 {len(samples)} samples  "
          f"reproj=[{min(rp):.3f}..{max(rp):.3f}]  "
          f"corners=[{min(cn)}..{max(cn)}]")
    print(f"🎯 TARGET: pos<{args.target_pos_mm}mm  rot<{args.target_rot_deg}°")
    print(f"{'='*70}")

    tp, tr, mn = args.target_pos_mm, args.target_rot_deg, args.min_samples
    all_r = []

    for name, fn in [("Reproj filter", strat_reproj),
                     ("Corner filter", strat_corners),
                     ("Combined", strat_combined),
                     ("Angle range", strat_angle),
                     ("Outlier removal", strat_outlier)]:
        print(f"\n{'─'*70}\nStrategy: {name}\n{'─'*70}")
        res = fn(samples, mn); all_r.extend(res)
        for r in sorted(res, key=lambda x:x['pos_mm'])[:6]:
            h = "🎯" if r['pos_mm']<tp and r['rot_deg']<tr else "  "
            print(f"  {h} {r['method']:12s} n={r['n']:2d} "
                  f"pos={r['pos_mm']:.3f}mm rot={r['rot_deg']:.3f}°  "
                  f"{r['filter']}")

    all_r.sort(key=lambda x: x['pos_mm'])
    print(f"\n{'='*70}\n🏆 TOP 10\n{'='*70}")
    for i, r in enumerate(all_r[:10]):
        h = "🎯" if r['pos_mm']<tp and r['rot_deg']<tr else "  "
        print(f"  {h} {i+1:2d}. {r['method']:12s} n={r['n']:2d} "
              f"pos={r['pos_mm']:.3f}mm rot={r['rot_deg']:.3f}°  "
              f"{r['filter']}")

    best = all_r[0]
    print(f"\n{'='*70}\n🏆 BEST: {best['method']}  "
          f"pos={best['pos_mm']:.3f}mm  rot={best['rot_deg']:.3f}°  "
          f"n={best['n']}")
    trans = best['T_tcp_cam'][:3,3]
    euler = R.from_matrix(best['T_tcp_cam'][:3,:3]).as_euler('xyz',degrees=True)
    print(f"   Translation: [{trans[0]:.4f}, {trans[1]:.4f}, {trans[2]:.4f}]m")
    print(f"   Euler:       [{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}]°")

    intr_file = 'results/intrinsics/camera_intrinsics.json'
    hit = save_best(best, intr_file, tp, tr)
    print(f"\n  {'🎯 TARGET ACHIEVED!' if hit else '⚠️  Target not achieved'}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()