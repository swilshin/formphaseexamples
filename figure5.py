
from linearformphase.formphaseutil import FourierSeries
from linearformphase import LinearFormPhase,LinearLocalFormPhase
from linearformphase import (toPolar,fromPolar,polarJacobian,
  contravariantFromPolar,covariantFromPolar,covariantToPolar,
  contravariantToPolar)
from linearformphase.examples import (rdotwinfree,rdotsimp,rdotselkov,rdotfhn,
  npsimple,wrapdrdt)
from linearformphase.rectification import rectification

from scipy.interpolate import interp1d
from scipy.integrate import odeint

from copy import deepcopy

set_printoptions(precision=3)

################################################################################
# INTEGRATE ODE
################################################################################
N0 = 10000 # 1000
Nt = 1
Ncyc = 3
mod = 'fhn' #'fhn', 'selkov', 'simple', 'winfree'
t = linspace(0,Ncyc*2.*pi,Nt)
method = 'polynomial' # 'local' 'polynomial'

if mod == 'winfree':
  R0 = 0.4
  rdot = rdotwinfree
  C = 1.0
elif mod == 'selkov':
  R0 = 0.2
  rdot = rdotselkov
  C = (2.*pi)/3.098640493202466
elif mod == "fhn":
  R0 = 0.1
  rdot = rdotfhn
  C = (2*pi)/(3.*3.743)
else:
  R0 = 0.5
  rdot = rdotsimp
  C = 1.0

tols=1e-10

################################################################################
# BUILD LIMIT CYCLE MODEL
################################################################################
tend = 100.
Ncyc = 5000
tcyc = linspace(0.,2*pi/C,Ncyc+1)[:-1]
thetacyc = tcyc*C
r0 = odeint(
  rdot,array([1,0]),[0.,tend],args=(),
  mxstep=100000,rtol=tols,atol=tols
)[-1]
rlcyc = odeint(
  rdot,r0,tcyc,args=(),
  mxstep=100000,rtol=tols,atol=tols
)
rlcyc[:,1] = rlcyc[:,1]%(2.*pi)
xshift0 = (
  array([-0.9,0.6])
    if mod=="fhn" else
  array([0.0,0.0])
)
xlcyc = fromPolar(rlcyc)-xshift0

rect = rectification(
  toPolar(fromPolar(rlcyc)-xshift0),
  ordR=20,ordP=20
)

thetasrct = array([rlcyc[...,0],linspace(0,2*pi,Ncyc)]).T
urecttheta = interp1d(
  hstack([rect(thetasrct)[...,1]-2*pi,rect(thetasrct)[...,1],rect(thetasrct)[...,1]+2*pi]),
  hstack([thetasrct[...,1]-2*pi,thetasrct[...,1],thetasrct[...,1]+2*pi])
)
# Form a radial grid after radial rectification
r0 = array([1.0+R0*(2*rand(N0)-1.0),2.*pi*rand(N0)]).T
r0[...,1] = urecttheta(r0[...,1])
r0[...,0]*=rect.fsR0.val(r0[...,1]).real[...,0]
r0 = toPolar(fromPolar(r0)+xshift0)

'''
r0 = array([1.0+R0*(2*rand(N0)-1.0),2.*pi*rand(N0)]).T
r0[...,0] = r0[...,0]*rect.fsR0.val([r0[...,1]])[:,0].real
'''

r = zeros((N0,Nt,2))
r[:,0] = r0
for i in xrange(N0):
  r[i] = odeint(
    rdot,r[i,0],t,args=(),mxstep=100000,rtol=tols,atol=tols
  )

dr = rdot(r)

ylcyc = fromPolar(rlcyc)-xshift0
rlcyc = toPolar(ylcyc)
dx = contravariantFromPolar(r,dr)
r0 = toPolar(fromPolar(r0)-xshift0)
x = fromPolar(r)-xshift0

r = toPolar(x)
dr = contravariantToPolar(x,dx)

# Rectify
plcyc = rect(rlcyc)
qlcyc = fromPolar(plcyc)

p = rect(r)
q = fromPolar(p)
J = rect.jacobian(r)
dp = einsum('...jk,...k->...j',J,dr)
dq = contravariantFromPolar(p,dp)

Nr = 6
ordP = 15

lph = LinearFormPhase()
lph.train(q,dq,C=C,Nr=Nr,order=ordP,orientation=-1)

# Manually compute the PRC
# Naive phase interpolant
from scipy.interpolate import interp1d
rlcycOC = toPolar(fromPolar(rlcyc)+xshift0)
pintrp = interp1d(
  hstack([rlcycOC[...,1]-2.*pi,rlcycOC[...,1],rlcycOC[...,1]+2.*pi]),
  hstack([thetacyc-2.*pi,thetacyc,thetacyc+2*pi])
)

eps = 1.e-3
Nstep = 50
prcr = list()
for i,ir in enumerate(rlcyc[::Nstep]):
  #print 'Processing {0} of {1}\r'.format(i+1,rlcyc.shape[0]/Nstep),
  ir0 = toPolar(fromPolar(ir)+xshift0)
  rf = odeint(
    rdot,ir0,[0,tend],args=(),
    mxstep=100000,rtol=tols,atol=tols
  )[-1,1]%(2.*pi)
  rp0 = odeint(
    rdot,ir0+[eps,0],[0,tend],args=(),
    mxstep=100000,rtol=tols,atol=tols
  )[-1,1]%(2.*pi)
  rp1 = odeint(
    rdot,ir0+[0,eps],[0,tend],args=(),
    mxstep=100000,rtol=tols,atol=tols
  )[-1,1]%(2.*pi)
  prcr.append((
    ((pintrp(rp0%(2.*pi))-pintrp(rf%(2*pi))+pi)%(2.*pi)-pi)/eps,
    ((pintrp(rp1%(2.*pi))-pintrp(rf%(2*pi))+pi)%(2.*pi)-pi)/eps
  ))
  print 'Proc {0} of {1}, rp0 - {2}, rp1 - {3}, rf - {4}, pirp0 - {5}, pir1 - {6}, pir - {7}, {8} & {9}'.format(
    i+1,rlcyc.shape[0]/Nstep,rp0,rp1,rf,
    pintrp(rp0%(2.*pi)),pintrp(rp1%(2.*pi)),pintrp(rf%(2*pi)),
    prcr[-1][0],prcr[-1][1]
  )
print ""

Jlcyc = rect.jacobian(rlcyc)
invJlcyc = array([inv(iJ) for iJ in Jlcyc])
#prcForm = covariantFromPolar(plcyc,lph.form(qlcyc))
prcForm = covariantFromPolar(
  rlcyc,
  einsum('...kj,...k->...j',Jlcyc,lph.form(qlcyc))
)
rr = vstack([rlcycOC[::Nstep],[rlcycOC[0]]])
pr = vstack([array(prcr),[prcr[0]]])
prcNumeric = covariantFromPolar(rr,pr)

data = [thetacyc,prcForm,prcNumeric,xlcyc,Nstep]

from pickle import dump

fle = open("prcdata.pkl",'w')
dump(data,fle)
fle.close()

figure()
plot(thetacyc,prcForm, c='b',alpha=0.7,lw=4)
plot(thetacyc,prcForm[:,0], c='k',alpha=0.9,lw=1)
plot(thetacyc,prcForm[:,1], c='w',alpha=0.9,lw=1)
plot(
  hstack([thetacyc[::Nstep],2.*pi]),
  prcNumeric, c='r',alpha=0.7,lw=4
)
plot(
  hstack([thetacyc[::Nstep],2.*pi]),
  prcNumeric[:,0], c='k',alpha=0.7,lw=1
)
plot(
  hstack([thetacyc[::Nstep],2.*pi]),
  prcNumeric[:,1], c='w',alpha=0.7,lw=1
)
xlabel("phase (rad)")
ylabel("phase response (rad)")
xlim(0,2*pi)
xticks(linspace(0,2*pi,5),["$0$","$\\pi/2$","$\\pi$","$3\\pi/3$","$2\\pi$"])
savefig("fhnprc.png")
savefig("fhnprc.pdf")
savefig("fhnprc.svg")
savefig("fhnprc.png")
