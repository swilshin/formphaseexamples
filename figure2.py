'''
Code for producing figure 2 of manuscript.
'''

import formphase

phi = formphase.LinearFormPhase()
phi.orientation = 1.0
phi.Nr = 5
phi.order = 5
N = 44
eta = 0.001
cmc = cm.jet # cm.gist_gray # cm.jet # cm.plasma
Nc = 20

x = linspace(-1.3,1.3,100)
X,Y = meshgrid(x,x)
XX = array([X,Y]).transpose(1,2,0)

phi.K = zeros(N)
phi0 = phi(XX)

r = formphase.toPolar(XX)
rho = r[...,0]-1.
theta = r[...,1]
z = r[...,2:]

figure(figsize=(16,8))
subplot(4,12,1)
contourf(X,Y,phi0,linspace(-pi,pi,Nc+1,endpoint=True),cmap=cmc)
contour(X,Y,phi0,linspace(-pi,pi,5,endpoint=False),colors='k',linestyles='solid',linewidths=1)
axis('off')
axis('equal')
for i in range(N):
  subplot(4,12,i+2+int(i/11))
  phi.K = zeros(N)
  phi.K[i] = eta
  Z = (phi(XX) - phi0)
  contourf(X,Y,Z,Nc+1,cmap=cmc)
  contour(X,Y,Z,5,colors='k',linestyles='solid',linewidths=1)
  axis('off')
  axis('equal')
tight_layout()
savefig('formphaseterms.png')

i = 2
figure()
phi.K = zeros(N)
phi.K[i] = eta
Z = phi(XX)  - phi0
contourf(X,Y,Z,100)
