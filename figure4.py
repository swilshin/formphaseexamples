from linearformphase.examples import rdotselkov
from linearformphase.formphaseutil import eventPhase
from linearformphase.formphaseutil import Phaser
from linearformphase.formphaseutil import FourierSeries

from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist

from pickle import load

with open("standardformselkov_polynomial_0.1_500_500_3.0_foriso.pkl",'r') as fle:
  data = load(fle)

with open("standardformselkov_polynomial_0.0_500_500_3.0_foriso.pkl",'r') as fle:
  data_nonoise = load(fle)

Ncontour = 20
Vctrs = linspace(0,2*pi,Ncontour+1,endpoint=True)
vtol = 0.0025

ph = data[15]
yd = data[9]
ylcyc = data[17]
rlcyc = data[18]
phiQu = data[4]
Xu = data[0][0]
Ru = data[0][1]

cols = {
  'phaser':'k',
  'form':'#106AA4', #Teal
  'free':'#106AA4', #Teal
  'event':'#43BF3C', # Green
  'true':'#FF7F00', # Orange
  'phaserl':"#404040",
  'eventl':'#83FF7C', # Green
}

# Ground Truth
t = [0,100.]
loadgt = True
if loadgt:
  with open("phgt.pkl",'r') as fle:
    phgt = load(fle)
else:
  phgt = []
  for ir in Ru:
    phgt.append([])
    for iir in ir:
      phgt[-1].append(odeint(rdotselkov,iir,t,args=(),mxstep=100000)[1][1])
    print "Another 1%"
phgt = array(phgt)
from pickle import dump
with open("phgt.pkl",'w') as fle:
  dump(phgt,fle)

phgt = phgt%(2.*pi)
plcyc = linspace(0,2.*pi,ylcyc.shape[0],endpoint=False)
pc = interp1d(
  hstack([rlcyc[...,1]-2.*pi,rlcyc[...,1],rlcyc[...,1]+2.*pi]),
  hstack([plcyc-2.*pi,plcyc,plcyc+2.*pi])
)
phgt = pc(phgt)

# Event
phe = array([eventPhase(iyd,n=[0.,-1.]) for iyd in yd])

# Phaser
phsr = Phaser(
  [iyd for iyd in yd],
  psecfunc=lambda x: x[1],
  ordP=20,
  protophfun=lambda x: array([x[0]+1.j*x[1]]*x.shape[0])
)
php = array([phsr(iyd)[0] for iyd in yd])

# Pick a cardinal point
cpidx = sum(
  abs(Xu-ylcyc[((ylcyc[...,0]<0.)
   +
  abs(ylcyc[...,1])).argmin()]),
  axis=-1
).argmin()
phiQu -= (
  phiQu.reshape((phiQu.size,))[cpidx]
   -
  phgt.reshape((phgt.size,))[cpidx]
)
phiQu %= 2.*pi

# Plot
for ph0 in [0.,pi]:
  figure(figsize=(12,6))
  ax = list()
  for i in xrange(3):
    if i==0:
      ax.append(subplot(1,3,i+1))
    else:
      ax.append(subplot(1,3,i+1,sharex=ax[0],sharey=ax[0]))
    for iyd in yd:
      q = sum(cdist(ylcyc,iyd.T).min(axis=0))/15.
      ax[-1].plot(*iyd,c='b',alpha=0.0001+0.0024*q)
    ax[-1].plot(*ylcyc.T,c='w',lw=3)
    ax[-1].plot(*ylcyc.T,c='k',lw=2)
    ax[-1].set_axis_off()

  # Add isochrone from form phase
  ax[2].contour(
    Xu[...,0],Xu[...,1],(phiQu+ph0)%(2*pi),
    Vctrs,colors=cols['form'],zorder=5
  )

  # Add isochrones from event phase
  darken = 1
  for vc in Vctrs[:-1]:
    yrng = yd.transpose(0,2,1)[logical_and(
      (phe-vc-vtol+pi)%(2*pi)-pi<0.,
      (phe-vc+vtol+pi)%(2*pi)-pi>0.
    )]
    ax[0].scatter(
      *yrng.T,
      color=cols['event'] if darken==1 else cols['eventl'],marker='.',
      s=9
    )
    ax[0].scatter(*mean(yrng,axis=0),marker='x',c='m',zorder=7)  
    darken*=-1

  # Add isochrones from phaser
  darken = 1
  for vc in Vctrs[:-1]:
    yrng = yd.transpose(0,2,1)[logical_and(
      (php-vc-vtol+pi)%(2*pi)-pi<0.,
      (php-vc+vtol+pi)%(2*pi)-pi>0.
    )]
    ax[1].scatter(
      *yrng.T,
      color=cols['phaser'] if darken==1 else cols['phaserl'],marker='.',s=9
    )
    ax[1].scatter(*mean(yrng,axis=0),marker='x',c='m',zorder=7)
    darken*=-1

  # Add ground truth
  for i in xrange(3):
    ax[i].contour(
      Xu[...,0],Xu[...,1],(phgt+ph0)%(2.*pi),
      Vctrs,colors=cols['true'],zorder=0
    )

  # Add limit cycle estimate
  fse = FourierSeries().fit(20,phe.reshape((phe.size,)),yd.transpose(0,2,1).reshape((ph.size,-1)).T)
  theta = linspace(0,2*pi,1000)
  ax[0].plot(*fse.val(theta).real.T,c='c',zorder=9)

  fse = FourierSeries().fit(20,php.reshape((php.size,)),yd.transpose(0,2,1).reshape((ph.size,-1)).T)
  theta = linspace(0,2*pi,1000)
  ax[1].plot(*fse.val(theta).real.T,c='c',zorder=9)

  fse = FourierSeries().fit(20,ph.reshape((ph.size,)),yd.transpose(0,2,1).reshape((ph.size,-1)).T)
  theta = linspace(0,2*pi,1000)
  ax[2].plot(*fse.val(theta).real.T,c='c',zorder=9)

  ax[0].set_xlim([-.4,0.95])
  ax[0].set_ylim([-.9,0.9])

  tight_layout()

  savefig("isocompselkov_"+str(ph0)+".png")
  savefig("isocompselkov_"+str(ph0)+".svg")

figure(figsize=(12,6))
lwmask = 6
subplot(1,3,1)
for iyd in yd:
  plot(*iyd,lw=lwmask,c='k')
axis('off')
xlim([-.4,0.95])
ylim([-.9,0.9])

subplot(1,3,2)
for iyd in yd:
  plot(*iyd,lw=lwmask,c='k')
axis('off')
xlim([-.4,0.95])
ylim([-.9,0.9])

subplot(1,3,3)
for iyd in yd:
  plot(*iyd,lw=lwmask,c='k')
axis('off')
xlim([-.4,0.95])
ylim([-.9,0.9])

tight_layout()

savefig("isocompselkov-mask.png")
savefig("isocompselkov-mask.svg")
