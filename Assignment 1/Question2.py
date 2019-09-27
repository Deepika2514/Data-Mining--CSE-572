# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 15:12:03 2019

@author: Deepika
"""

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(num=None, figsize=(12, 6), dpi=80, facecolor='w')
import matplotlib
import pandas as pd


df=pd.read_csv(r"Downloads\overdoses.csv")

df["Deaths"]=df["Deaths"].str.replace(',','')
df["Population"]=df["Population"].str.replace(',','')

df["Deaths"]=df["Deaths"].apply(int)
df["Population"]=df["Population"].apply(int)

df1={"ODD":df.Deaths/df.Population,"State":df.Abbrev}

df["Deaths"]=df["Deaths"].apply(int)
df["Population"]=df["Population"].apply(int)

df2=pd.DataFrame(df1)

plt.bar(df2.State,df2.ODD)
plt.ylabel("Opioid Death Density")
plt.xlabel('State Codes')
plt.xticks(rotation='vertical')
plt.title('Bar Graph for ODD vs States')
plt.savefig('barchart.png')



