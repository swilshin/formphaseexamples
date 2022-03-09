from linearformphase.formphaseutil import FourierSeries

from numpy import (array,linspace,pi,hstack,vstack,arctan2,sqrt,sum,logical_and,
  nan)

from pylab import (cm,figure,subplot,contour,contourf,plot,axis,savefig,
  tight_layout)

from pickle import load
import argparse

if __name__=="__main__":
  ################################################################################
  # PROCESS COMMAND LINE
  ################################################################################
  parser = argparse.ArgumentParser(description='Fit a form phase estimator to a system of differential equations.')
  parser.add_argument(
    '-s','--system',
    type=str,
    help='Which system to fit, options are fhn, selkov, simple, winfree',
    default="winfree"
  )
  parser.add_argument(
    '-m','--method',
    type=str,
    help='Which type of form phase to fit, options are polynomial or local',
    default="polynomial"
  )
  parser.add_argument(
    '-n','--noise',
    type=float,
    help='noise level',
    default=0.0
  )
  parser.add_argument(
    '-l','--length',
    type=int,
    help='The length of the simulation runs',
    default=100
  )
  parser.add_argument(
    '-r','--runs',
    type=int,
    help='The number of the simulation runs',
    default=200
  )
  parser.add_argument(
    '-c','--cycles',
    type=float,
    help='The number of the simulation runs',
    default=3.0
  )
  parser.add_argument(
    '-N','--Note',
    type=str,
    help='Add a note to the file name',
    default=""
  )

  args = parser.parse_args()
  mod = args.system
  method = args.method
  sigmaNoise = args.noise #0.1 # 0.1
  N0 = args.runs
  Nt = args.length
  Ncyc = args.cycles
  note = args.Note
  
  if note=="":
    fn = "standardform"+mod+"_"+method+"_"+str(sigmaNoise)+"_"+str(N0)+"_"+str(Nt)+"_"+str(Ncyc)+".pkl"
  else:
    fn = "standardform"+mod+"_"+method+"_"+str(sigmaNoise)+"_"+str(N0)+"_"+str(Nt)+"_"+str(Ncyc)+"_"+note+".pkl"
  
  # Pickle in data used to make plots
  fle = open(fn,'r')

  Cu,C,Cr,phiPu,phiQu,phiA,phiLX,phiR,phiX,yd,rd,ud,vd,pd,qd,ph,phiTerms,ylcyc,rlcyc,ulcyc,vlcyc,plcyc,qlcyc,misc,desc = load(fle)

  Xu,Ru,Au,Bu,Pu,Qu = Cu
  X,R,A,B,P,Q = C
  Xr,Rr,Ar,Br,Pr,Qr = Cr

  # Set up co-ordinates
  Nctr = 101
  pad = 0.4
  theta0 = 0.
  phi0 = 0.0
  lalpha = 0.4
  Ncontour = 50
  Nisoc = 50
  Vctrs = linspace(0,2*pi,Ncontour+1,endpoint=True)
  lcol = 'w'
  phishift = pi # 0, pi
  lcyccol = '#009090'

  from matplotlib.cm import datad
  from matplotlib.colors import LinearSegmentedColormap
  cdict = dict([(k,sorted([((x[0]+0.5)%1.,)+x[1:] for x in datad['hsv'][k]],key=lambda x: x[0])) for k in datad['hsv']])
  edgec = dict([(k,(array(cdict[k][0])+array(cdict[k][-1]))[1:]/2) for k in cdict])
  cdict = dict([(k,vstack([hstack([0.,edgec[k]]),cdict[k],hstack([1.,edgec[k]])])) for k in cdict])
  hsv_shifted = LinearSegmentedColormap('hsv_shifted',cdict)
  if (phishift == 0.):
    ctrcm = cm.hsv
  else:
    ctrcm = hsv_shifted

  subplots = False

  def makeRadius(y,Nth=100,Nf=20):
    y = array(y)
    tau,rho = hstack(arctan2(*y.transpose(1,0,2))), hstack(sqrt(sum((y*y),axis=1)))
    thetas = linspace(-pi,pi,Nth)
    rmax = FourierSeries().fit(
      Nf,
      array([(thetas[:-1]+thetas[1:])/2.]),
      array([[rho[logical_and(tau>t0,tau<t1)].max() for t0,t1 in zip(thetas[:-1],thetas[1:])]])
    )
    rmin = FourierSeries().fit(
      Nf,
      array([(thetas[:-1]+thetas[1:])/2.]),
      array([[rho[logical_and(tau>t0,tau<t1)].min() for t0,t1 in zip(thetas[:-1],thetas[1:])]])
    )
    return(rmin,rmax)

  ydmin,ydmax = makeRadius(yd)
  vdmin,vdmax = makeRadius(vd)
  qdmin,qdmax = makeRadius(qd)

  # Fix phases outside of data
  def trimPhases(X,xmax,xmin,ph,pad=0.1):
    idxOut = sqrt(sum(X*X,axis=-1))>xmax.val(arctan2(*X.transpose(2,0,1))).real[:,0].reshape(X.shape[:-1])+pad
    idxIn = sqrt(sum(X*X,axis=-1))<xmin.val(arctan2(*X.transpose(2,0,1))).real[:,0].reshape(X.shape[:-1])-pad
    ph[idxOut] = nan #arctan2(*X.transpose(2,0,1))[idxOut]%(2*pi)
    ph[idxIn] = nan# arctan2(*X.transpose(2,0,1))[idxIn]%(2*pi)
    return(ph)

  phiPu = (trimPhases(Xu,ydmax,ydmin,phiPu) + phishift)%(2.*pi)
  phiQu = (trimPhases(Xu,ydmax,ydmin,phiQu) + phishift)%(2.*pi)
  phiA = (trimPhases(X,vdmax,vdmin,phiA) + phishift)%(2.*pi)
  phiR = (trimPhases(Xr,qdmax,qdmin,phiR) + phishift)%(2.*pi)
  phiLX = (trimPhases(X,qdmax,qdmin,phiLX) + phishift)%(2.*pi)
  phiX = (trimPhases(Xr,qdmax,qdmin,phiX) + phishift)%(2.*pi)

  # Plot
  if subplots:
    figure(figsize=(16,10))
    subplot(2,3,1)
  else:
    figure(figsize=(10,10))
  contourf(Xu[...,0],Xu[...,1],phiPu,Vctrs,cmap=ctrcm)
  contour(Xu[...,0],Xu[...,1],phiPu,Vctrs,colors='k',zorder=5)
  for iyd in yd:
    plot(*iyd,c=lcol,alpha=lalpha)
  plot(*ylcyc.T,c='w',lw=3,zorder=6)
  plot(*ylcyc.T,c=lcyccol,lw=2,zorder=7)
  axis('off')
  if not subplots:
    savefig("lineartermcontributionplot"+"ps-"+("0" if phishift==0. else "pi")+"-mod-"+mod+"-meth-"+method+"_1.svg")

  if subplots:
    subplot(2,3,6)
  else:
    figure(figsize=(10,10))
  contourf(Xu[...,0],Xu[...,1],phiQu,Vctrs,cmap=ctrcm)
  contour(Xu[...,0],Xu[...,1],phiQu,Vctrs,colors='k',zorder=5)
  for iyd in yd:
    plot(*iyd,c=lcol,alpha=lalpha)
  plot(*ylcyc.T,c='w',lw=3,zorder=6)
  plot(*ylcyc.T,c=lcyccol,lw=2,zorder=7)
  axis('off')
  if not subplots:
    savefig("lineartermcontributionplot"+"ps-"+("0" if phishift==0. else "pi")+"-mod-"+mod+"-meth-"+method+"_6.svg")

  if subplots:
    subplot(2,3,2)
  else:
    figure(figsize=(10,10))
  contourf(X[...,0],X[...,1],phiA,Vctrs,cmap=ctrcm)
  contour(X[...,0],X[...,1],phiA,Vctrs,colors='k',zorder=5)
  for ivd in vd:
    plot(*ivd,c=lcol,alpha=lalpha)
  plot(*vlcyc.T,c='w',lw=3,zorder=6)
  plot(*vlcyc.T,c=lcyccol,lw=2,zorder=7)
  axis('off')
  axis('equal')
  if not subplots:
    savefig("lineartermcontributionplot"+"ps-"+("0" if phishift==0. else "pi")+"-mod-"+mod+"-meth-"+method+"_2.svg")

  if subplots:
    subplot(2,3,4)
  else:
    figure(figsize=(10,10))
  contourf(X[...,0],X[...,1],phiLX,Vctrs,cmap=ctrcm)
  contour(X[...,0],X[...,1],phiLX,Vctrs,colors='k',zorder=5)
  for iqd in qd:
    plot(*iqd,c=lcol,alpha=lalpha)
  plot(*qlcyc.T,c='w',lw=3,zorder=6)
  plot(*qlcyc.T,c=lcyccol,lw=2,zorder=7)
  axis('off')
  axis('equal')
  tight_layout(pad=0.05, w_pad=0.05, h_pad=0.05)
  if not subplots:
    savefig("lineartermcontributionplot"+"ps-"+("0" if phishift==0. else "pi")+"-mod-"+mod+"-meth-"+method+"_4.svg")

  if subplots:
    subplot(2,3,3)
  else:
    figure(figsize=(10,10))
  contourf(Xr[...,0],Xr[...,1],phiR,Vctrs,cmap=ctrcm)
  contour(Xr[...,0],Xr[...,1],phiR,Vctrs,colors='k',zorder=5)
  for iqd in qd:
    plot(*iqd,c=lcol,alpha=lalpha)
  plot(*qlcyc.T,c='w',lw=3,zorder=6)
  plot(*qlcyc.T,c=lcyccol,lw=2,zorder=7)
  axis('off')
  axis('equal')
  if not subplots:
    savefig("lineartermcontributionplot"+"ps-"+("0" if phishift==0. else "pi")+"-mod-"+mod+"-meth-"+method+"_3.svg")

  if subplots:
    subplot(2,3,5)
  else:
    figure(figsize=(10,10))
  contourf(Xr[...,0],Xr[...,1],phiX,Vctrs,cmap=ctrcm)
  contour(Xr[...,0],Xr[...,1],phiX,Vctrs,colors='k',zorder=5)
  for iqd in qd:
    plot(*iqd,c=lcol,alpha=lalpha)
  plot(*qlcyc.T,c='w',lw=3,zorder=6)
  plot(*qlcyc.T,c=lcyccol,lw=2,zorder=7)
  axis('off')
  axis('equal')
  if subplots:
    tight_layout(pad=0.05, w_pad=0.05, h_pad=0.05)
    savefig("lineartermcontributionplot"+"ps-"+("0" if phishift==0. else "pi")+"-mod-"+mod+"-meth-"+method+".png")
    savefig("lineartermcontributionplot"+"ps-"+("0" if phishift==0. else "pi")+"-mod-"+mod+"-meth-"+method+".svg")
  else:
    savefig("lineartermcontributionplot"+"ps-"+("0" if phishift==0. else "pi")+"-mod-"+mod+"-meth-"+method+"_5.svg")
