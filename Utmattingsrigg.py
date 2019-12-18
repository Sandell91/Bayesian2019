
# coding: utf-8

# # Analystiska beräkningar för utmattningsrigg
# 
# Skiss från min anteckningsbok:
# 
# ![Frilaggning](Frilaggning.png)
# 
# Enligt ASTME855-08 så ska avsåndet mellan belastningspunkterna vara 1/6-del av längden. Det går inte i vårt fall p.g.a begränsningar i lastriggen.
# Detta dokument ska utröna vilka dimensioner som kan vara rimliga.
# 
# Stavarnas längd är 180 mm. Thomas föreslog minst 10 mm  säkerhetsmarginal i varje ände d.v.s maxlängd=160mm
# Avståndet mellan inre och yttre kontaktpunkt anges i % av L där 0<A<0.5L
# Höjd och bredd anges i mm. Initialt med startvärdena 15 mm
# Kraften som riggen klarar av att leverera är 20 kN. För att ha lite säkerhetsmarginal räknar vi med en 18 kN last.
# Målspänningen är 675 MPa (det som användes i Georgssons böjprovning)

# In[29]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
import itertools
import pandas as pd


# # Calculations

# In[48]:


L=np.arange(0.09,.160,.0001) #Längd mellan yttre kontaktpunkterna 90<L<160 mm
a=np.arange(0,.5,.5/np.size(L))#Avstånd mellan yttre och inre kontaktpunkt i % av L. 0<a<0.5L om a=0.5L så har vi trepunktsböj.
h=0.015 #Höjd i m
b=0.015 #Bredd i m
E=115e9 #E-modul, normalt sett 110-117 MPa


F=18000 #Maxlast i N

x=[]
for i in L:
    np.array(x.append(np.arange(0,i,i/len(L)))) #Distance vector for making plots.
x=np.array(x)#transform list to array

A=0; B=L[-1]*a; C=L[-1]*(1-a); D=L[-1] #Defines the contact points ABCD

## Region AB
V_AB=F/2 #Inre sjuvkraft i region AB
Mz_AB=F*x/2 #Moment 

## Region BC
V_BC=0 #Inre sjuvkraft i region AB
Mz_BC=L[-1]*a*F/2 #Moment med avseende på Z i region BC

## Region CD
V_CD=-F/2 #Inre sjuvkraft i region AB
Mz_CD=-F*x/2 #Moment med avseende på Z i region BC

y=-h/2 #position i y för utvärdering av spänning och töjning i ytan
I_z=b*h**3/12 #Yttröghetsmoment med avseende på z


sigma_max=-Mz_BC*y/I_z #spänning i region BC
Defle=F*L[-1]*a*(3*L[-1]**2-4*(L[-1]*a)**2)/(48*E*I_z) #Nedböjning under last F
epsilon_surf=sigma_max/E #Töjning i ytan under dragspänning

print(r'I trepunktsböj får vi: delta_max='+str(np.round(Defle[-1]*1000,2))+'[mm] and sigma_max='+str(np.round(sigma_max[-1]/10e6,2))+' [MPa]')


# # Plotting

# In[46]:


Intersect=np.argmin(np.abs(sigma_max-675e6))
Avst=1000*a*L[-1] #Enhetsbyte till mm

#Main plots
plt.figure(figsize=(15,10))
plt.plot((Avst[0],Avst[-1]),(675e6,675e6),'--')
plt.axvline(x=Avst[Intersect], color='k')
plt.axvline(x=1000*L[-1]/6, color='k', alpha=.5)
plt.plot(Avst,sigma_max)


#Plot text in the figure
plt.text(Avst[Intersect], 0, ' Intersect='+str(np.round(1000*L[-1]*a[Intersect],4))+' mm',{'color': 'k'}, fontsize=25,ha='left')
plt.text(1000*L[-1]/6, 0, 'ASTM E855-08  ',{'color': 'k'}, fontsize=15,ha='right',alpha=.5)
plt.text(0, 680e6, r'$\sigma=675$ MPa',{'color': 'tab:blue'}, fontsize=25,ha='left',va='bottom')
textBOX = '\n'.join((
    r'$b=%.1f$ mm' % (1000*b, ),
    r'$h=%.1f$ mm' % (1000*h, ),
    r'$L=%.f$ mm' % (1000*L[-1], ),
    r'$E=%i$ GPa' % (E*10**(-9), )))        
plt.text(0, np.max(sigma_max),textBOX,fontsize=20,va='top')
plt.text(np.max(Avst), 400e6, r'$\delta_{intersect}=%.3f$ mm' %(1000*Defle[Intersect]),{'color': 'k'}, fontsize=20,ha='right',va='bottom')
plt.text(np.max(Avst), 400e6, r'$\epsilon_{surf, intersect}=%.5f$' %(epsilon_surf[Intersect]),{'color': 'k'}, fontsize=20,ha='right',va='top')


#Axis labels
plt.xlabel('Distance between outer and inner contact points (a*L) [mm]',fontsize=30)
plt.ylabel('Stress [Pa]',fontsize=30)
plt.show()


# In[47]:


Intersect=np.argmin(np.abs(sigma_max-675e6))
Avst=1000*a*L[-1]
plt.figure(figsize=(15,10))
#plt.plot((Avst[0],Avst[-1]),(1.762,1.762),'--')
plt.axvline(x=Avst[Intersect], color='k')
plt.plot(Avst,Defle*1000)
plt.plot(Avst[Intersect], Defle[Intersect]*1000,'ro')

#Plots text in the figure
plt.text(Avst[Intersect], 0, ' Intersect='+str(np.round(1000*L[-1]*a[Intersect],4))+' mm',{'color': 'k'}, fontsize=25,ha='left')
plt.text(Avst[Intersect], Defle[Intersect]*1000, r'   $\delta_{intersect}=%.3f$ mm' %(1000*Defle[Intersect]),{'color': 'r'}, fontsize=20,ha='left',va='top')

#Axis labels
plt.xlabel('Distance between contact points [mm]',fontsize=30)
plt.ylabel(r'Deflection ($\delta$) [mm]',fontsize=30)
plt.show()


# In[20]:


Defle[-1]*1000


# In[23]:


sigma_max[-1]/10e6

