
from __future__ import print_function

from linearformphase.examples.metadata import BirdData

from numpy import (array,sum,isnan,hstack,diff,logical_xor,arange,sqrt,dot,
  mean,einsum,linspace,meshgrid,pi,unwrap,cov)
from numpy.linalg import eig
from numpy.random import seed,randint
from scipy.interpolate import UnivariateSpline

from linearformphase import LinearFormPhase
from util import FourierSeries

from os import mkdir
from os.path import join as osjoin
from os.path import isdir

from pylab import (figure,plot,subplot,scatter,Rectangle,contour,contourf,axis,
  xlim,ylim,xticks,yticks,tight_layout,savefig,hexbin,Circle)

################################################################################
# Parameters
################################################################################
cols = {
  'phaser':'k',
  'form':'#106AA4', #Teal
  'event':'#43BF3C', # Green
  'true':'#FF7F00' # Orange
}

mDPI = 600
dirs = ["X","Y","Z"]
birds = ["GRY","RED","G00","YEL","NOC","RRE"]

method = "integral" # "differential"

df = 0.5 # Fraction of data to use
ws = 20 # Window size

figDir = "./figure"
if not isdir(figDir):
  mkdir(figDir)
levelPerturb = 1

seed(0)

################################################################################
# Load the bird data
################################################################################
brdData = dict()
for bird in birds:
  baseDir = osjoin(".","exampledata","gfforceplate","2013-1120-"+bird)
  brdData[bird] = BirdData(baseDir)
  # Keep only the level trials
  brdData[bird].dat = [d for d in brdData[bird] if d.meta.perturbType==levelPerturb]

################################################################################
# Chop into short sections
################################################################################
txs = dict()
dtxs = dict()
gaps = list()
for bird in birds:
  txs[bird] = list()
  b = brdData[bird]
  for t in b.dat:
    for side in ["Left","Right"]:
      # Cut into segments without nans
      idxnan = (sum(isnan((t.x[side+"_digit_3"]-t.x.CoM)[:,[0,2]]),axis=1)>0)
      idxbound = [] if idxnan[0] else [0]
      idxbound = hstack([idxbound,arange(1,idxnan.size)[logical_xor(idxnan[1:],idxnan[:-1])]])
      idxbound = hstack([idxbound,[] if idxnan[-1] else [idxnan.size-1]])
      # Loop over continuous segments without nans
      for i,j in zip(idxbound[:-1],idxbound[1:]):
        if sum(isnan((t.x[side+"_digit_3"]-t.x.CoM)[i:j,[0,2]]))==0 and j>i+ws:
          tx = (t.x[side+"_digit_3"]-t.x.CoM)[i:j,[0,2]].T
          # Grab random segments ws in length
          N = tx.shape[1]
          idxt = array(sorted(randint(0,N,int(df*N/ws)))+[N-1])
          idxn = [idxt[0]]
          for i in idxt[1:-1]:
            if i-idxn[-1]>ws and i+ws+1<N:
              idxn.append(i)
          for i in idxn:
            if tx[:,i:i+ws].shape[1]==ws:
              txs[bird].append(tx[:,i:i+ws])
              gaps.append(j-i+ws)
  txs[bird] = array(txs[bird])
  dtxs[bird] = diff(txs[bird])
  # dtxs[bird] = array([[UnivariateSpline(arange(itrl.size),itrl).derivative()(arange(itrl.size)) for itrl in trl] for trl in txs[bird].transpose(1,2,0)]).transpose(2,0,1)
  txs[bird] = (txs[bird][:,:,:-1]+txs[bird][:,:,:-1])/2.

  x0 = hstack(txs[bird])
  dx0 = hstack(dtxs[bird])
  # z-score and PCA
  D,U = eig(cov(x0))
  x = (sqrt(1./D)*dot(U.T,x0).T).T
  mux = mean(x,axis=1)
  dx = (sqrt(1./D)*dot(U.T,dx0).T).T
  x = (x.T - mux).T
  txs[bird] = einsum("i,jik->jik",sqrt(1./D),einsum("jl,ijk->ilk",U,txs[bird]))
  dtxs[bird] = einsum("i,jik->jik",sqrt(1./D),einsum("jl,ijk->ilk",U,dtxs[bird]))
  txs[bird] = (txs[bird].transpose(0,2,1) - mux).transpose(0,2,1)
print("Min gap size:", min(gaps))
print("Number of cycles:", sum([t.x.data.shape[0] for t in brdData[bird] for bird in birds])/60)

cutthresh = -0.5
x = hstack([hstack(txs[bird]) for bird in birds])
dx = hstack([hstack(dtxs[bird]) for bird in birds])
idxright = x[0]>cutthresh
Nrb = 6 # 10, 6, 4
Nf = 6 # 10, 6, 4
if True:
  phi = LinearFormPhase()
  if method=="differential":
    phi.train(x.T,dx.T,C=None,Nr=Nrb,order=Nf,orientation=-1.,usecg=False)
  else:
    phi.trainx(x.T,dx.T,C=None,Nr=Nrb,order=Nf,orientation=-1.,usecg=False)
  '''phicut = formPhase(
    x[:,idxright],
    dx[:,idxright],
    ordPLim=20,
    Nsb=200,
    minR=0.001
  )'''

  phicut = LinearFormPhase()
  if method=="differential":
    phicut.train(x[:,idxright].T,dx[:,idxright].T,C=None,Nr=Nrb,order=Nf,orientation=-1.,usecg=False)
  else:
    phicut.trainx(x[:,idxright].T,dx[:,idxright].T,C=None,Nr=Nrb,order=Nf,orientation=-1.,usecg=False)

  ################################################################################
  # Limit cycle model
  ################################################################################
  p = phi(x.T)
  pcut = phicut(x.T)
  f = FourierSeries().fit(10,p,x)
  fcut = FourierSeries().fit(10,pcut,x)

  ################################################################################
  # Chop into short sections
  ################################################################################
  figure(figsize=(12,12))
  for bird in birds:
    for tx,dtx in zip(txs[bird],dtxs[bird]):
      plot(*tx,c='b',lw=2,alpha=0.6)

  # Since the contour plot is not aware of the topology of the oscillator it 
  # generate a thick black line at the discontinuity in our co-ordinate map. 
  # To address this we create a left and a right version of the figure which 
  # have this discontinuity on opposite sides.
  # Left
  Ncs = 400
  Ncont = 41 # Must be odd
  pad = 0.05
  xL = x.min(axis=1)
  xH = x.max(axis=1)
  X0 = linspace(xL[0]-pad,xH[0]+pad,Ncs)
  Y0 = linspace(xL[1]-pad,xH[1]+pad,Ncs)
  X,Y = meshgrid(X0,Y0)
  Z = array([X.reshape(X.size),Y.reshape(Y.size)])
  pC = phi(Z.T)
  pD = phicut(Z.T)

  # Make pD agree with pC via correction term
  thetalcyc = linspace(0,2*pi,100)
  lcyc = f.val(thetalcyc).real.T
  lcyccut = fcut.val(thetalcyc).real.T
  #pD -= mean((phi(lcyc[:,lcyc[0]>cutthresh])-phicut(lcyc[:,lcyc[0]>cutthresh])+pi)%(2*pi)-pi)
  #fscor = FourierSeries().fit(20,phi(lcyc),array([(phicut(lcyc)-phi(lcyc)+pi)%(2.*pi)-pi]))
  #pD = pD - fscor.val(pD).real[:,0]

  s = UnivariateSpline(
    hstack([
      unwrap(phi(f.val(linspace(0,2*pi,100,endpoint=False)).real))-2.*pi,
      unwrap(phi(f.val(linspace(0,2*pi,100,endpoint=False)).real)),
      unwrap(phi(f.val(linspace(0,2*pi,100,endpoint=False)).real))+2.*pi,
    ]),
    hstack([
      unwrap(phicut(f.val(linspace(0,2*pi,100,endpoint=False)).real))-2.*pi,
      unwrap(phicut(f.val(linspace(0,2*pi,100,endpoint=False)).real)),
      unwrap(phicut(f.val(linspace(0,2*pi,100,endpoint=False)).real))+2.*pi,
    ]),
    s=0.
  )


  # Make these agree at a fixed point on the limit cycle
  #pD = (pD + phi(array([lcyc[:,0]]).T) - phicut(array([lcyc[:,0]]).T))%(2.*pi)

  figure()
  plot(phicut(lcyc.T))
  plot(phi(lcyc.T))
  plot(phi(lcyccut.T))
  plot(phicut(lcyccut.T))

  from scipy.stats import gaussian_kde

  k = gaussian_kde(hstack([hstack(txs[bird]) for bird in birds]))
  dens = k(Z)

  concols = ['#FF7F00','#FFBF90','#FFDFD0']
  ncontcolour = '#007F6F'
  alphabackground = 0.5

  fig = figure(figsize=(36,12))
  ax0 = subplot(1,3,1)
  for bird in birds:
    for tx in txs[bird]:
      scatter(*tx,c='b',lw=2,alpha=0.6,zorder=1,marker='x')
      scatter(*tx[:,tx[0]>cutthresh],c='k',lw=2,alpha=0.3,zorder=0,marker='o') # Thinner high alpha line + add high alpha data in orange / yellow + mask
  plot(*lcyc,lw=3,c='k',zorder=2)
  plot(*txs["NOC"][170],lw=8,c='k',zorder=5,alpha=1.)
  plot(*txs["NOC"][170][:,txs["NOC"][170][0]>cutthresh],lw=6,c='b',zorder=6,alpha=1.)
  plot(*txs["NOC"][170][:,txs["NOC"][170][0]>cutthresh],lw=3,c='w',zorder=7,alpha=1.)
  contourf(X,Y,dens.reshape(X.shape)/dens.max(),[.1,0.25,0.5,1.0],zorder=0,colors=concols,alpha=alphabackground)
  ax0.add_patch(
    Rectangle(
      (-5,-5),
      5+cutthresh,
      10,
      facecolor="w",
      alpha=0.3,
      edgecolor="k",
      linewidth=3,
      zorder=8
    )
  )
  axis('equal')
  xlim((x.min(axis=1)[0],x.max(axis=1)[0]))
  ylim((x.min(axis=1)[1],x.max(axis=1)[1]))
  xticks([])
  yticks([])
  axis('off')
  ax1 = subplot(1,3,2)
  #contour(X,Y,(pD.reshape(X.shape))%(2*pi),s(linspace(0,2*pi,Ncont))%(2.*pi),colors='r',linewidths=3,zorder=1)
  contour(X,Y,(pC.reshape(X.shape))%(2*pi),linspace(0,2*pi,Ncont),colors=cols['form'],linewidths=3,zorder=2)
  contourf(X,Y,dens.reshape(X.shape)/dens.max(),[.1,0.25,0.5,1.0],zorder=0,colors=concols,alpha=alphabackground)
  contourf(X,Y,dens.reshape(X.shape)/dens.max(),[0.0,0.1],colors='w',zorder=5)
  #plot(*lcyccut,lw=3,c='g',zorder=4,alpha=0.8)
  plot(*lcyc,lw=3,c='k',zorder=3,alpha=0.8)
  ax1.add_patch(
    Rectangle(
      (-5,-5),
      5+cutthresh,
      10,
      facecolor="w",
      alpha=0.3,
      edgecolor="k",
      linewidth=3,
      zorder=5
    )
  )
  axis('equal')
  xlim((x.min(axis=1)[0],x.max(axis=1)[0]))
  ylim((x.min(axis=1)[1],x.max(axis=1)[1]))
  xticks([])
  yticks([])
  axis('off')
  ax2 = subplot(1,3,3)
  contour(
    X,Y,(pD.reshape(X.shape))%(2*pi),
    sorted(list(set((s(linspace(0,2*pi,Ncont)+pi)-pi)%(2.*pi))))[:-1],
    colors=ncontcolour,linewidths=3,zorder=2
  )
  contour(X,Y,(pC.reshape(X.shape))%(2*pi),
    linspace(0,2*pi,Ncont),
    colors=cols['form'],linewidths=3,zorder=1,alpha=0.5
  )
  contourf(X,Y,dens.reshape(X.shape)/dens.max(),
    [.1,0.25,0.5,1.0],zorder=0,colors=concols,alpha=alphabackground
  )
  #contour(X,Y,dens.reshape(X.shape)/dens.max(),[0.5,0.25,0.1],zorder=6,linewidths=3,colors='k')
  contourf(X,Y,dens.reshape(X.shape)/dens.max(),
    [0.0,0.1],colors='w',zorder=5
  )
  plot(*lcyccut,lw=3,c=cols['event'],zorder=3,alpha=1.0)
  plot(*lcyc,lw=3,c='k',zorder=4,alpha=0.5)
  ax2.add_patch(
    Rectangle(
      (-5,-5),
      5+cutthresh,
      10,
      facecolor="w",
      alpha=0.9,
      edgecolor="k",
      linewidth=3,
      zorder=5
    )
  )
  axis('equal')
  xlim((x.min(axis=1)[0],x.max(axis=1)[0]))
  ylim((x.min(axis=1)[1],x.max(axis=1)[1]))
  xticks([])
  yticks([])
  axis('off')
  tight_layout()
  savefig('cutperformancecompR.png')

  fig = figure(figsize=(36,12))
  ax0 = subplot(1,3,1)
  for bird in birds:
    for tx in txs[bird]:
      scatter(*tx,c='b',lw=2,alpha=0.6,zorder=1,marker='x')
      scatter(*tx[:,tx[0]>cutthresh],c='k',lw=2,alpha=0.3,zorder=0,marker='o') # Thinner high alpha line + add high alpha data in orange / yellow + mask
  plot(*lcyc,lw=3,c='k',zorder=2)
  plot(*txs["NOC"][170],lw=8,c='k',zorder=5,alpha=1.)
  plot(*txs["NOC"][170][:,txs["NOC"][170][0]>cutthresh],lw=6,c='b',zorder=6,alpha=1.)
  plot(*txs["NOC"][170][:,txs["NOC"][170][0]>cutthresh],lw=3,c='w',zorder=7,alpha=1.)
  contourf(X,Y,dens.reshape(X.shape)/dens.max(),[.1,0.25,0.5,1.0],zorder=0,colors=concols,alpha=alphabackground)
  ax0.add_patch(
    Rectangle(
      (-5,-5),
      5+cutthresh,
      10,
      facecolor="w",
      alpha=0.3,
      edgecolor="k",
      linewidth=3,
      zorder=8
    )
  )
  axis('equal')
  xlim((x.min(axis=1)[0],x.max(axis=1)[0]))
  ylim((x.min(axis=1)[1],x.max(axis=1)[1]))
  xticks([])
  yticks([])
  axis('off')
  ax1 = subplot(1,3,2)
  #contour(X,Y,(pD.reshape(X.shape)+pi)%(2*pi),(s(linspace(0,2*pi,Ncont)+pi)-pi)%(2.*pi),colors='r',linewidths=3,zorder=1)
  contour(X,Y,(pC.reshape(X.shape)+pi)%(2*pi),linspace(0,2*pi,Ncont),colors=cols['form'],linewidths=3,zorder=2)
  #contour(X,Y,dens.reshape(X.shape)/dens.max(),[0.5,0.25,0.1],zorder=6,linewidths=3,colors='k')
  contourf(X,Y,dens.reshape(X.shape)/dens.max(),[.1,0.25,0.5,1.0],zorder=0,colors=concols,alpha=alphabackground)
  contourf(X,Y,dens.reshape(X.shape)/dens.max(),[0.0,0.1],colors='w',zorder=5)
  #plot(*lcyccut,lw=3,c='g',zorder=4,alpha=0.8)
  plot(*lcyc,lw=3,c='k',zorder=3,alpha=0.8)
  ax1.add_patch(
    Rectangle(
      (-5,-5),
      5+cutthresh,
      10,
      facecolor="w",
      alpha=0.3,
      edgecolor="k",
      linewidth=3,
      zorder=5
    )
  )
  axis('equal')
  xlim((x.min(axis=1)[0],x.max(axis=1)[0]))
  ylim((x.min(axis=1)[1],x.max(axis=1)[1]))
  xticks([])
  yticks([])
  axis('off')
  ax2 = subplot(1,3,3)
  contour(
    X,Y,(pD.reshape(X.shape)+pi)%(2*pi),
    sorted(list(set((s(linspace(0,2*pi,Ncont)+pi)-pi)%(2.*pi))))[:-1],
    colors=ncontcolour,linewidths=3,zorder=2
  )
  contour(X,Y,(pC.reshape(X.shape)+pi)%(2*pi),linspace(0,2*pi,Ncont),colors=cols['form'],linewidths=3,zorder=1,alpha=0.5)
  #contour(X,Y,dens.reshape(X.shape)/dens.max(),[0.5,0.25,0.1],zorder=6,linewidths=3,colors='k')
  contourf(X,Y,dens.reshape(X.shape)/dens.max(),[.1,0.25,0.5,1.0],zorder=0,colors=concols,alpha=alphabackground)
  contourf(X,Y,dens.reshape(X.shape)/dens.max(),[0.0,0.1],colors='w',zorder=5)
  plot(*lcyccut,lw=3,c=cols['event'],zorder=3,alpha=1.0)
  plot(*lcyc,lw=3,c='k',zorder=4,alpha=0.5)
  ax2.add_patch(
    Rectangle(
      (-5,-5),
      5+cutthresh,
      10,
      facecolor="w",
      alpha=0.9,
      edgecolor="k",
      linewidth=3,
      zorder=5
    )
  )
  axis('equal')
  xlim((x.min(axis=1)[0],x.max(axis=1)[0]))
  ylim((x.min(axis=1)[1],x.max(axis=1)[1]))
  xticks([])
  yticks([])
  axis('off')
  tight_layout()
  savefig('cutperformancecompL.png')

  fig = figure(figsize=(36,12))
  ax0 = subplot(1,3,1)
  axis('equal')
  xlim((x.min(axis=1)[0],x.max(axis=1)[0]))
  ylim((x.min(axis=1)[1],x.max(axis=1)[1]))
  xticks([])
  yticks([])
  axis('off')
  ax1 = subplot(1,3,2)
  for bird in birds:
    for tx,dtx in zip(txs[bird],dtxs[bird]):
      plot(*tx,c='k',lw=50,alpha=1)
  axis('equal')
  xlim((x.min(axis=1)[0],x.max(axis=1)[0]))
  ylim((x.min(axis=1)[1],x.max(axis=1)[1]))
  xticks([])
  yticks([])
  axis('off')
  ax2 = subplot(1,3,3)
  for bird in birds:
    for tx,dtx in zip(txs[bird],dtxs[bird]):
      plot(*tx,c='k',lw=50,alpha=1)
  axis('equal')
  xlim((x.min(axis=1)[0],x.max(axis=1)[0]))
  ylim((x.min(axis=1)[1],x.max(axis=1)[1]))
  xticks([])
  yticks([])
  axis('off')
  tight_layout()
  savefig('cutperformancecompmask.png')

  p0 = phi.form.getForm(x.T)
  p0cut = phicut.form.getForm(x.T)
  figure()
  hexbin(x[0],x[1],abs(sum(p0*x,axis=0)-phi.form.C),100)
  figure()
  hexbin(x[0],x[1],abs(sum(p0cut*x,axis=0)-phicut.form.C),100)

  # Plot
  figure(figsize=(24,12))
  subplot(1,2,1)
  for bird in birds:
    for tx in txs[bird]:
      scatter(*tx,c='b',lw=2,alpha=0.6,zorder=0,marker='x')
  plot(*lcyc,lw=3,c='k',zorder=2)
  plot(*txs["NOC"][0],lw=8,c='k',zorder=5,alpha=1.)
  plot(*txs["NOC"][0],lw=6,c='b',zorder=6,alpha=1.)
  plot(*txs["NOC"][0],lw=3,c='w',zorder=7,alpha=1.)
  axis('equal')
  xlim((x.min(axis=1)[0],x.max(axis=1)[0]))
  ylim((x.min(axis=1)[1],x.max(axis=1)[1]))
  xticks([])
  yticks([])
  axis('off')
  tight_layout()
  subplot(1,2,2)
  contour(X,Y,pC.reshape(X.shape),linspace(0,2*pi,Ncont),colors=cols['form'],linewidths=3,zorder=1)
  contour(X,Y,dens.reshape(X.shape)/dens.max(),[0.5,0.25,0.1],zorder=6,linewidths=3,colors='k')
  contourf(X,Y,dens.reshape(X.shape)/dens.max(),[0.0,0.1],colors='w',zorder=5)
  plot(*lcyc,lw=3,c='k',zorder=2)
  axis('equal')
  xlim((x.min(axis=1)[0],x.max(axis=1)[0]))
  ylim((x.min(axis=1)[1],x.max(axis=1)[1]))
  xticks([])
  yticks([])
  axis('off')
  tight_layout()
  savefig("choppedturkeyL.svg")
  savefig("choppedturkeyL.png")

  # Right
  figure(figsize=(24,12))
  subplot(1,2,1)
  for bird in birds:
    for tx,dtx in zip(txs[bird],dtxs[bird]):
      scatter(*tx,c='b',lw=2,alpha=0.6,zorder=0,marker='x')
  plot(*lcyc,lw=3,c='k',zorder=2)
  plot(*txs["NOC"][0],lw=8,c='k',zorder=5,alpha=1.)
  plot(*txs["NOC"][0],lw=6,c='b',zorder=6,alpha=1.)
  plot(*txs["NOC"][0],lw=3,c='w',zorder=7,alpha=1.)
  axis('equal')
  xlim((x.min(axis=1)[0],x.max(axis=1)[0]))
  ylim((x.min(axis=1)[1],x.max(axis=1)[1]))
  xticks([])
  yticks([])
  axis('off')
  tight_layout()
  subplot(1,2,2)
  contour(X,Y,(pC.reshape(X.shape)+pi)%(2*pi),linspace(0,2*pi,Ncont),colors=cols['form'],linewidths=3,zorder=1)
  contour(X,Y,dens.reshape(X.shape)/dens.max(),[0.5,0.25,0.1],zorder=6,linewidths=3,colors='k')
  contourf(X,Y,dens.reshape(X.shape)/dens.max(),[0.0,0.1],colors='w',zorder=5)
  plot(*lcyc,lw=3,c='k',zorder=2)
  axis('equal')
  xlim((x.min(axis=1)[0],x.max(axis=1)[0]))
  ylim((x.min(axis=1)[1],x.max(axis=1)[1]))
  xticks([])
  yticks([])
  axis('off')
  tight_layout()
  savefig("choppedturkeyR.svg")
  savefig("choppedturkeyR.png")

  # A mask which will allow us to mask the isochones where it is likely they are 
  # inaccurate to avoid misleading the reader about our interpolants abilities 
  # outside of the region where we have training data.
  figure(figsize=(24,12))
  subplot(1,2,1)
  for bird in birds:
    for tx,dtx in zip(txs[bird],dtxs[bird]):
      plot(*tx,c='k',lw=50,alpha=1)
  axis('equal')
  xlim((x.min(axis=1)[0],x.max(axis=1)[0]))
  ylim((x.min(axis=1)[1],x.max(axis=1)[1]))
  xticks([])
  yticks([])
  axis('off')
  tight_layout()
  subplot(1,2,2)
  for bird in birds:
    for tx,dtx in zip(txs[bird],dtxs[bird]):
      plot(*tx,c='k',lw=50,alpha=1)
  axis('equal')
  xlim((x.min(axis=1)[0],x.max(axis=1)[0]))
  ylim((x.min(axis=1)[1],x.max(axis=1)[1]))
  xticks([])
  yticks([])
  axis('off')
  tight_layout()
  savefig("choppedturkeymask.svg")
  savefig("choppedturkeymask.png")

  fig = figure()
  ax = fig.add_subplot(1,1,1)
  for c,s in zip(phi.form.c,phi.form.s):
    cir = Circle(c,s,facecolor='r',edgecolor='r',lw=2,alpha=0.1)
    ax.add_artist(cir)
  for c,s in zip(phicut.form.c,phicut.form.s):
    cir = Circle(c,s,facecolor='b',edgecolor='b',lw=2,alpha=0.1)
    ax.add_artist(cir)
  axis('equal')
  xlim((x.min(axis=1)[0],x.max(axis=1)[0]))
  ylim((x.min(axis=1)[1],x.max(axis=1)[1]))
  xticks([])
  yticks([])
  axis('off')
  tight_layout()
