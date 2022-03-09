from formphase.hmaposc import (FloquetCoordinateOsc,FloquetCoordinateSim,
  FloquetCoordinateOscFactory,HMapFactory,HMap,HMapChainFactory,HMapChain,
  HFloquetSystem,HFloquetSystemFactory,HFloquetSim,SimSpec,Noise,Times,
  saveSimulation,DataSetFactory,loadSimulation,FMap,FMapFactory)
from formphase.formphaseutil import (EllipseReflectorFactory,EllipseReflector,
  kalman,Kalman,PCA,serialiseArray,unserialiseArray,FourierSeries,RTrans)
from formphase.formphaseutil import paddedHilbert,eventPhase,circleMean
from formphase.hmaposc import plots
from formphase.formphaseutil import Phaser

from linearformphase import (toPolar,fromPolar,contravariantToPolar,
  contravariantFromPolar)
from linearformphase import LinearFormPhase,rectification

from gzip import open as gzopen
from cPickle import dump

import matplotlib
from scipy.signal import hilbert
from numpy import (hstack,std,min,max,sqrt,abs,unwrap,angle,pi,dot,cos,sin,
  einsum,ones,linspace,array,zeros)
from numpy.random import seed

from os import listdir
from os.path import join as osjoin
from os.path import split as ossplit

if __name__=='__main__':
  seed(0)
  
  Ntrain = 50
  angl = +pi/4
  Ra = RTrans([[cos(angl),sin(angl)],[-sin(angl),cos(angl)]])
  
  matplotlib.rcParams.update({'font.size': 24})
  
  #===========================================================================
  # LOAD SIMULATIONS
  #===========================================================================  
  # Have the user select which system they want to analyse
  # Set directories the simulations are in
  simDirs = [
    osjoin("sim","2D"),
    osjoin("sim","3D"),
    osjoin("sim","8D")
  ]

  # Grab list of files and pick candidates with the same system in them
  sims = list()
  for d in simDirs:
    fls = [
      f for f in listdir(d)
      if f[-7:]==".pkl.gz" and f[-13:]!='_cache.pkl.gz'
    ]
    for f in fls:
      try:
        s = loadSimulation(osjoin(d,f))
        if (
          (
            sims==[] # Simulation list is empty
              or
            not ( # Not one we've found
              repr(eval(s[0]).sim) in zip(*zip(*sims)[1])[0]
                and
              repr(eval(s[0]).spec) in zip(*zip(*sims)[1])[1]
            )
          )  
            and
          f[:-7]+"_kal.py" in listdir(d) # Has Kalman filter          
        ):
          sims.append(
            (osjoin(d,f),
            (repr(eval(s[0]).sim),repr(eval(s[0]).spec)))
          )
      except EOFError:
        print f, "is in the simulation folder, has the right extension but does not seem to be a valid simulation."

  # Have the user select one of these simulations to analyse
  options = range(len(sims))+[-1]
  simfls = zip(*sims)[0]+("all",)
  print "Found the following files with Kalman filters."
  for idx,f in zip(options,simfls):
    print str(idx)+":",f
  s = None
  while s is None:
    try:
      print "Please select simulation to analyse"
      s = int(raw_input("$ "))
    except:
      s = None # If the user input is invalid in any way, ask again
    if not s in options:
      s = None
  
  if s==-1:
    lst = options[:-1]
  else:
    lst = [s]
  
  for s in lst:
    # Assemble data, if system is 2D we do not use PCA, if it is 8D we use 
    # finite differences to calculate derivatives
    simf = DataSetFactory(simfls[s],eval(zip(*sims)[1][s][0]).D!=2,False)
      
    sim = simf(Ntrain)
    
    #===========================================================================
    # TRAIN ALGORITHMS
    #===========================================================================
    print "Training form phase..."
    '''phi = formPhase(
      sim.trials.yt,
      minR=0.01,ordPLim=25,fourCor2=True,rbfPen=0.0,orientation=None,Nsb=100,minSig=0.2 # Nsb=250, Nsb=500, 100 for non-final versions works faster
    )'''
    Nlcyc = 1000
    Norder = 100
    rholcyc = array([
      ones((Nlcyc,)),
      linspace(0,2*pi,Nlcyc),
    ]).T
    rholcyc = hstack([rholcyc,zeros((Nlcyc,sim.H.sim.D-2))])
    xlcyc = fromPolar(rholcyc)
    ylcyc = sim.trials.dtran(sim.H.sim.Hsys(xlcyc).T).T
    rlcyc = toPolar(ylcyc)
    rect = rectification(rlcyc,Norder)
    
    if sim.H.sim.D==8:
      q = [fromPolar(rect(toPolar(iyd.T))) for iyd in sim.trials.yd.getX()]
      dq = hstack([diff(iq.T) for iq in q]).T
      q = hstack([iq[:-1].T for iq in q]).T
    else:
      x = sim.trials.yd.getFlatX().T
      dx = sim.trials.yd.getFlatdX().T
      r = toPolar(x)
      dr = contravariantToPolar(x,dx)
      # Rectify
      p = rect(r)
      q = fromPolar(p)
      D,U = eig(cov(q.T))
      J = rect.jacobian(r)
      dp = einsum('...jk,...k->...j',J,dr)
      dq = contravariantFromPolar(p,dp)
    
    phi = LinearFormPhase()
    phi.train(q,dq,C=None,Nr=6,order=12,orientation=-1. if sim.H.sim.D==2 else 1.) # C=2.*pi
    
    ordP=50 if sim.H.sim.D!=3 else 10 # Going higher causes ringing
    cycTol = 0.05
    print "Training Phaser..."
    # Stop phaser trying to train on things it cant
    gdTrials = [
      min(sqrt(sum(iyt**2,0)))>0.1
      and
      max(sqrt(sum(iyt**2,0)))<100.0 for iyt in sim.trials.yt.getX()
    ]
    cycs = sum(unwrap(angle(hilbert(sim.trials.yt.getX(),axis=2)),2)[:,:,-1]/(2*pi),0)
    cycs = abs(cycs-cycs[0])/cycs[0]<cycTol
    print "Training on: ", sum(gdTrials), "out of: ", len(gdTrials)

    psi = Phaser(
      [iyt[cycs][:,200:] for iyt,gd in zip(sim.trials.yt.getX(),gdTrials) if gd],
      ordP=ordP,psecfunc=lambda x: dot(sim.psf[cycs],x),protophfun=paddedHilbert if sum(cycs)>=2 else hilbert
    )

    #===========================================================================
    # ESTIMATE PHASES
    #===========================================================================
    print "Calculating phases..." 
    ph = {
      'phaser':[unwrap(psi(iyd[cycs])[0]) for iyd in sim.trials.yd.getX()],
      'form':[unwrap(phi(fromPolar(rect(toPolar(iyd.T))))) for iyd in sim.trials.yd.getX()],
      'event':[unwrap(eventPhase(iyd,n=sim.psf)) for iyd in sim.trials.yd.getX()],
      'true':[ix[:,0] for ix in sim.trials.xd]
    }
    # Phase differences
    phR = dict([
      (k,[iph-ipht for iph,ipht in zip(ph[k],ph['true'])])
      for k in ph
    ])
    phRMu = dict([(k,circleMean(hstack(phR[k]))) for k in phR])
    # Shift so all phases are aligned
    ph = dict([(k,(unwrap(ph[k]-phRMu[k])%(2*pi))) for k in ph])
    phR = dict([(k,(unwrap(phR[k]-phRMu[k]+pi)%(2*pi)-pi)) for k in phR])
    
    print 
    
    print "Writing cache..."
    with gzopen(simf.noExtSource+"_cache.pkl.gz",'wb') as fle:
      dump([repr(sim),repr(phi),ph,phR],fle,-1)

    #===========================================================================
    # PLOTS
    #===========================================================================
    # Plots are organised in to generic plots and plots that only function for 
    # two three and eight dimensional systems.
    #===========================================================================
    if True:
      from pylab import figure,savefig
      from matplotlib.colors import LinearSegmentedColormap
      
      cols = {
        'phaser':'k',
        'form':'#106AA4', #Teal
        'free':'#106AA4', #Teal
        'event':'#43BF3C', # Green
        'true':'#FF7F00' # Orange
      }
      cmaps = {
        'phaser':LinearSegmentedColormap.from_list('phaser cmap',[cols['phaser'],cols['phaser'],'#808080']),
        'form':LinearSegmentedColormap.from_list('form cmap',[cols['form'],cols['form'],'#808080']),
        'free':LinearSegmentedColormap.from_list('free cmap',[cols['free'],cols['free'],'#808080']),
        'event':LinearSegmentedColormap.from_list('event cmap',[cols['event'],cols['event'],'#808080']),
        'true':LinearSegmentedColormap.from_list('true cmap',[cols['true'],cols['true']]),
      }
      markers = {
        'phaser':'s',
        'form':'D',
        'event':'h',
        'true':'+'
      }    
      
      print "Plotting results..."
      
      errN = 3.0*min([std(phR['event']),std(phR['form']),std(phR['phaser'])])
      Nstep = 500
      fig = figure(figsize=(16,7))
      ax=fig.add_subplot(131)
      plots.plotPhases(phR,ax=ax)
      ax=fig.add_subplot(132)
      plots.errorByPhase(ph,phR,ax=ax)
      ax=fig.add_subplot(133)
      h = plots.errorHistograms(phR,errN=errN,Nstep=Nstep,ax=ax)
      
      #print simfls[s].split("/")[1],float(fn.split("Sys")[1].split("_")[0]),float(fn.split("Init")[1].split("_")[0]),float(fn.split("Phase")[1].split("_")[0]),[std(phR[k]) for k in ['event','phaser','form']]
      
      '''
      Use the results from the histogram to calculate partial probability 
      plots for the phase error. More peaked means a better phase estimate.
      '''
      fig = figure()
      dP,idxP = plots.errorPartialProbPlot(
        1,1,1,h,phR,sim.trials.yd0,fig=fig,semilog=True,xl=(-errN,errN),yl=(0,None)
      )
      savefig(osjoin(
        "figures",
        ossplit(simf.noExtSource)[-1]+"_performance"+str(sim.H.sim.D)+".png")
      )
      savefig(osjoin(
        "figures",
        ossplit(simf.noExtSource)[-1]+"_performance"+str(sim.H.sim.D)+".svg")
      )
