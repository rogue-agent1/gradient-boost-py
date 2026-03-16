#!/usr/bin/env python3
"""Gradient boosting regressor with decision stumps."""
import random, sys

class Stump:
    def __init__(self): self.feat=0; self.thresh=0; self.left=0; self.right=0
    def fit(self, X, residuals):
        best=float('inf'); n=len(X)
        for f in range(len(X[0])):
            vals=sorted(set(row[f] for row in X))
            for i in range(len(vals)-1):
                t=(vals[i]+vals[i+1])/2
                l=[r for x,r in zip(X,residuals) if x[f]<=t]
                r_=[r for x,r in zip(X,residuals) if x[f]>t]
                if not l or not r_: continue
                ml,mr=sum(l)/len(l),sum(r_)/len(r_)
                mse=sum((ri-ml)**2 for ri in l)+sum((ri-mr)**2 for ri in r_)
                if mse<best: best=mse; self.feat=f; self.thresh=t; self.left=ml; self.right=mr
    def predict(self, x): return self.left if x[self.feat]<=self.thresh else self.right

class GradientBoosting:
    def __init__(self, n_estimators=50, lr=0.1):
        self.n_est=n_estimators; self.lr=lr; self.trees=[]; self.init=0
    def fit(self, X, y):
        self.init=sum(y)/len(y)
        preds=[self.init]*len(y)
        for _ in range(self.n_est):
            residuals=[y[i]-preds[i] for i in range(len(y))]
            stump=Stump(); stump.fit(X,residuals); self.trees.append(stump)
            for i in range(len(y)): preds[i]+=self.lr*stump.predict(X[i])
    def predict(self, X):
        return [self.init+sum(self.lr*t.predict(x) for t in self.trees) for x in X]

if __name__ == "__main__":
    random.seed(42)
    X=[[x/10] for x in range(100)]
    y=[3*x[0]**2-2*x[0]+1+random.gauss(0,0.5) for x in X]
    gb=GradientBoosting(n_estimators=100,lr=0.3); gb.fit(X,y)
    preds=gb.predict(X)
    mse=sum((p-t)**2 for p,t in zip(preds,y))/len(y)
    print(f"MSE: {mse:.4f}")
    for i in range(0,100,25): print(f"  x={X[i][0]:.1f}: pred={preds[i]:.2f} actual={y[i]:.2f}")
