import numpy as np, json, time, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT='results/c3'
os.makedirs(OUT,exist_ok=True)
d,r=100,5
m=5*d*r
eta=0.0025
T=20000
alphas=[1.0,1e-3]
seeds=[0,1,2]
LOG=100
iu=np.triu_indices(d)
diag_mask=(iu[0]==iu[1])
nu=len(iu[0])

def gen_W(rng):
    # weighted upper-triangle vectorization: <A,S> = W_i . S_upper for symmetric S
    W=np.empty((m,nu),dtype=np.float32)
    ch=250
    for s in range(0,m,ch):
        Q=rng.standard_normal((ch,d,d))
        A=0.5*(Q+np.transpose(Q,(0,2,1)))
        v=A[:,iu[0],iu[1]]
        v[:,~diag_mask]*=2.0
        W[s:s+ch]=v.astype(np.float32)
    return W

def up(S):
    return S[iu[0],iu[1]]

def M_from_g(g):
    # g = W^T r ; M = (1/m) sum r_i A_i (symmetric)
    Mv=g/m
    Mv=Mv.copy(); Mv[~diag_mask]/=2.0
    Mm=np.zeros((d,d)); Mm[iu[0],iu[1]]=Mv; Mm=Mm+Mm.T-np.diag(np.diag(Mm))
    return Mm

def gen_Xstar(rng):
    Us,_=np.linalg.qr(rng.standard_normal((d,r)))
    return Us@Us.T, Us   # spectral norm 1, kappa=1

# ---------------- positive controls ----------------
rngc=np.random.default_rng(123)
mc=400
Q=rngc.standard_normal((mc,d,d)); Ac=0.5*(Q+np.transpose(Q,(0,2,1)))
Wc=Ac[:,iu[0],iu[1]].copy(); Wc[:,~diag_mask]*=2
Xs_c,Us_c=gen_Xstar(rngc)
yc=Wc@up(Xs_c)
# (a) exact solution => zero train and test error
Uex=Us_c
tr_ex=0.5*np.mean((Wc@up(Uex@Uex.T)-yc)**2); te_ex=np.sum((Uex@Uex.T-Xs_c)**2)
# (b) <A,S> via W equals trace inner product
S=rngc.standard_normal((d,d)); S=S+S.T
ip_err=abs(Wc[0]@up(S)-np.sum(Ac[0]*S))
# (c) M reconstruction equals (1/m) sum r_i A_i
rr=rngc.standard_normal(mc)
M_err=np.max(np.abs(M_from_g(Wc.T@rr)*m/mc - np.tensordot(rr,Ac,axes=1)/mc))
# (d) test error = expected squared error on fresh measurements
U0=0.3*rngc.standard_normal((d,d))/np.sqrt(d)
D=U0@U0.T-Xs_c
fresh=np.einsum('kij,ij->k',Ac,D)  # these A are independent of U0
mc_ratio=np.mean(fresh**2)/np.sum(D**2)
control_pass=bool(tr_ex<1e-20 and te_ex<1e-20 and ip_err<1e-8 and M_err<1e-10 and abs(mc_ratio-1)<0.2)
del Ac,Q,Wc

# ---------------- main experiment ----------------
res={a:{'train':[], 'test':[]} for a in alphas}
curves={a:[] for a in alphas}
t0=time.time()
for sd in seeds:
    rng=np.random.default_rng(1000+sd)
    Xs,_=gen_Xstar(rng)
    W=gen_W(rng)
    y=(W@up(Xs).astype(np.float32)).astype(np.float64)
    B,_=np.linalg.qr(rng.standard_normal((d,d)))
    Us=[a*B for a in alphas]
    tr_c=[[] for _ in alphas]; te_c=[[] for _ in alphas]
    for t in range(T+1):
        Sst=np.stack([up(U@U.T) for U in Us],1).astype(np.float32)
        R=(W@Sst).astype(np.float64)-y[:,None]
        if t%LOG==0 or t==T:
            for k,U in enumerate(Us):
                tr_c[k].append(0.5*np.mean(R[:,k]**2))
                te_c[k].append(np.sum((U@U.T-Xs)**2))
        if t==T: break
        G=(W.T@R.astype(np.float32)).astype(np.float64)
        for k in range(len(Us)):
            Mm=M_from_g(G[:,k])
            Us[k]=Us[k]-eta*Mm@Us[k]
        if not np.all(np.isfinite(Us[0])): break
    for k,a in enumerate(alphas):
        res[a]['train'].append(tr_c[k][-1]); res[a]['test'].append(te_c[k][-1])
        curves[a].append((np.array(tr_c[k]),np.array(te_c[k])))
    del W
elapsed=time.time()-t0

Xnorm=float(r)  # ||X*||_F^2
tr1=float(np.mean(res[1.0]['train'])); te1=float(np.mean(res[1.0]['test']))
tr3=float(np.mean(res[1e-3]['train'])); te3=float(np.mean(res[1e-3]['test']))
cond_train=all(v<1e-4 for v in res[1.0]['train'])
cond_test=all(1e-2<=v<=1 for v in res[1.0]['test'])
cond_rel=all(1e-2<=v/Xnorm<=1 for v in res[1.0]['test'])
cond_cmp=all(b<a for a,b in zip(res[1.0]['test'],res[1e-3]['test']))

fig,ax=plt.subplots(1,2,figsize=(11,4))
its=np.arange(len(curves[1.0][0][0]))*LOG
for a,c in zip(alphas,['r','b']):
    for i,(trc,tec) in enumerate(curves[a]):
        ax[0].semilogy(its[:len(trc)],trc,c,alpha=.6,label=f'a={a}' if i==0 else None)
        ax[1].semilogy(its[:len(tec)],tec,c,alpha=.6,label=f'a={a}' if i==0 else None)
ax[0].set_title('train error f(U)'); ax[1].set_title('test error ||UU^T-X*||_F^2')
for x in ax: x.legend(); x.set_xlabel('iter')
plt.tight_layout(); plt.savefig(f'{OUT}/fig2_repro.png',dpi=100)

if not control_pass:
    status='inconclusive'; note='positive control failed'
elif cond_train and cond_test and cond_cmp:
    status='supported'; note='alpha=1 fits train <1e-4 with test in [1e-2,1]; alpha=1e-3 test lower'
elif cond_train and cond_rel and cond_cmp:
    status='inconclusive'; note='criterion met only for relative test error (/||X*||_F^2); absolute normalization of X* in paper unclear'
else:
    status='falsified'; note=f'cond_train={cond_train}, cond_test={cond_test}, cond_rel={cond_rel}, cond_cmp={cond_cmp}. Test err=||UU^T-X*||_F^2, X*=orthonormal rank-5 projector, symmetric Gaussian A, update U<-(I-eta M)U.'

summary={'claim_id':'C3','status':status,'metrics':{
 'train_alpha1_mean':tr1,'test_alpha1_mean':te1,'train_alpha1e-3_mean':tr3,'test_alpha1e-3_mean':te3,
 'train_alpha1_per_seed':res[1.0]['train'],'test_alpha1_per_seed':res[1.0]['test'],
 'test_alpha1e-3_per_seed':res[1e-3]['test'],'rel_test_alpha1_mean':te1/Xnorm,
 'cond_train':cond_train,'cond_test':cond_test,'cond_rel':cond_rel,'cond_cmp':cond_cmp,
 'control_pass':control_pass,'control_mc_ratio':float(mc_ratio),'elapsed_s':elapsed,'m':m,'T':T},
 'notes':note}
print('SUMMARY_JSON='+json.dumps(summary,default=str))
