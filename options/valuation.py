import pandas as pd
import numpy as np
import math as mt
from sklearn.linear_model import LinearRegression

def Binomial_Tree(Spot, Strike, Vencimiento, Volatilidad, TLibre_Riesgo, Call_Put, Tasa_Foranea=0, Tasa_Dividendo=0,
                  Ramificaciones_Arbol=100, Modelo="Cox Equity"):
    if Modelo == "Cox Equity":
        ConfigModelo = TLibre_Riesgo - Tasa_Dividendo
    if Modelo == "Cox Futuros":
        ConfigModelo = 0
    if Modelo == "Cox Divisas":
        ConfigModelo = TLibre_Riesgo - Tasa_Foranea

    Arbol_Subyacente = np.zeros((Ramificaciones_Arbol + 1, Ramificaciones_Arbol + 1))
    Arbol_Derivado = np.zeros((Ramificaciones_Arbol + 1, Ramificaciones_Arbol + 1))

    Vencimiento = Vencimiento / 365.0
    Steps = Vencimiento / Ramificaciones_Arbol
    Up = mt.exp(Volatilidad * mt.sqrt(Steps))
    Down = mt.exp(-Volatilidad * mt.sqrt(Steps))
    P = (mt.exp(ConfigModelo * Steps) - Down) / (Up - Down)

    # Obtener las ultimas ramas del arbol binomial del precio del subyacente
    Arbol_Subyacente[0, 0] = Spot

    for i in range(1, Ramificaciones_Arbol + 1):
        Arbol_Subyacente[i, 0] = Arbol_Subyacente[i - 1, 0] * Up
        for j in range(1, i + 1):
            Arbol_Subyacente[i, j] = Arbol_Subyacente[i - 1, j - 1] * Down

    for j in range(Ramificaciones_Arbol + 1):
        Arbol_Derivado[Ramificaciones_Arbol, j] = max(0,
                                                      Call_Put * (Arbol_Subyacente[Ramificaciones_Arbol, j] - Strike))

    for m in range(Ramificaciones_Arbol + 1):
        i = Ramificaciones_Arbol - m - 1
        for j in range(i + 1):
            Arbol_Derivado[i, j] = max(Call_Put * (Arbol_Subyacente[i, j] - Strike),
                                       (P * Arbol_Derivado[i + 1, j] + (1 - P) * Arbol_Derivado[i + 1, j + 1]) * mt.exp(
                                           -TLibre_Riesgo * Steps))

    # return pd.concat([pd.DataFrame(Arbol_Subyacente).replace(0,""),pd.DataFrame(Arbol_Derivado).replace(0,"")])
    return Arbol_Derivado[0, 0]


def Trinomial_Tree(Spot, Strike, Vencimiento, Volatilidad, TLibre_Riesgo, Call_Put, Tasa_Foranea=0, Tasa_Dividendo=0,
                   Ramificaciones_Arbol=100, Modelo="Cox Equity"):
    if Modelo == "Cox Equity":
        ConfigModelo = TLibre_Riesgo - Tasa_Dividendo
    if Modelo == "Cox Futuros":
        ConfigModelo = 0
    if Modelo == "Cox Divisas":
        ConfigModelo = TLibre_Riesgo - Tasa_Foranea

    Arbol_Subyacente = np.zeros((Ramificaciones_Arbol + 1, (2 * Ramificaciones_Arbol) + 1))
    Arbol_Derivado = np.zeros((Ramificaciones_Arbol + 1, (2 * Ramificaciones_Arbol) + 1))

    Vencimiento = Vencimiento / 365.0
    Steps = Vencimiento / Ramificaciones_Arbol
    Up = mt.exp(Volatilidad * mt.sqrt(2 * Steps))
    Down = mt.exp(-Volatilidad * mt.sqrt(2 * Steps))
    Pu = ((mt.exp(TLibre_Riesgo * Steps / 2) - mt.exp(-Volatilidad * mt.sqrt(Steps / 2))) / (
                mt.exp(Volatilidad * mt.sqrt(Steps / 2)) - mt.exp(-Volatilidad * mt.sqrt(Steps / 2)))) ** 2
    Pd = ((mt.exp(Volatilidad * mt.sqrt(Steps / 2)) - mt.exp(TLibre_Riesgo * Steps / 2)) / (
                mt.exp(Volatilidad * mt.sqrt(Steps / 2)) - mt.exp(-Volatilidad * mt.sqrt(Steps / 2)))) ** 2
    Pm = 1 - (Pu + Pd)

    # Obtener las ultimas ramas del arbol binomial del precio del subyacente
    Arbol_Subyacente[0, 0] = Spot

    for i in range(1, Ramificaciones_Arbol + 1):
        Arbol_Subyacente[i, 0] = Arbol_Subyacente[i - 1, 0] * Up
        for j in range(1, (2 * i)):
            Arbol_Subyacente[i, j] = Arbol_Subyacente[i - 1, j - 1]
            Arbol_Subyacente[i, j + 1] = Arbol_Subyacente[i - 1, j - 1] * Down

    for j in range((2 * Ramificaciones_Arbol) + 1):
        Arbol_Derivado[Ramificaciones_Arbol, j] = max(Call_Put * (Arbol_Subyacente[Ramificaciones_Arbol, j] - Strike),
                                                      0)

    for m in range(Ramificaciones_Arbol + 1):
        i = Ramificaciones_Arbol - m - 1
        for j in range((2 * i) + 1):
            Arbol_Derivado[i, j] = max(Call_Put * (Arbol_Subyacente[i, j] - Strike), (
                        Pu * Arbol_Derivado[i + 1, j] + Pm * Arbol_Derivado[i + 1, j + 1] + Pd * Arbol_Derivado[
                    i + 1, j + 2]) * mt.exp(-TLibre_Riesgo * Steps))

    # return pd.concat([pd.DataFrame(Arbol_Subyacente).replace(0,""),pd.DataFrame(Arbol_Derivado).replace(0,"")])
    return Arbol_Derivado[0, 0]

def LSM(Spot,Strike,Vencimiento,Volatilidad,TLibre_Riesgo,Call_Put,NumSim=10,CambiosXDia=1):

    Deltat = 1/(Vencimiento*CambiosXDia) # Asumo N Cambios en el precio del subyacente por cada día
    Caminos_Subyacente = np.zeros((NumSim,(Vencimiento*CambiosXDia)+1))
    v = Volatilidad/mt.sqrt(365/Vencimiento) # Se ajusta v pues v es anualizada
    r = TLibre_Riesgo/(365/Vencimiento) # Se ajusta r pues r es anualizada

    for m in range(0,NumSim):
        Caminos_Subyacente[m,0] = Spot
        for t in range(1,(Vencimiento*CambiosXDia)+1):
            Caminos_Subyacente[m,t] = Caminos_Subyacente[m,t-1]*mt.exp((r - (v**2)/2)*Deltat + np.random.normal(0,1)*mt.sqrt((v**2)*Deltat))

    Caminos_Derivado = np.zeros((NumSim,(Vencimiento*CambiosXDia)+1))
    Caminos_Derivado[:,(Vencimiento*CambiosXDia)] = np.maximum((Caminos_Subyacente[:,(Vencimiento*CambiosXDia)] - Strike)*Call_Put,0)

    for t in range((Vencimiento*CambiosXDia)-1,-1,-1):
        Caminos_Derivado[:,t] = Caminos_Derivado[:,t+1]*mt.exp(-r*Deltat) # Valor de Continuidad Observado (HV)
        Caminos_EnEl_Dinero = ((Caminos_Subyacente[:,t]-Strike)*Call_Put>0)
        if Caminos_EnEl_Dinero.sum()>0:
            Tabla_Regresion = np.zeros((Caminos_EnEl_Dinero.sum(),4))
            Tabla_Regresion[:,0] = Caminos_Subyacente[:,t][Caminos_EnEl_Dinero] #np.vectorize(mt.exp)(-Caminos_Subyacente[:,t][Caminos_EnEl_Dinero]/2)
            Tabla_Regresion[:,1] = Caminos_Subyacente[:,t][Caminos_EnEl_Dinero]**2 #np.vectorize(mt.exp)(-Caminos_Subyacente[:,t][Caminos_EnEl_Dinero]/2)*(1-Caminos_Subyacente[:,t][Caminos_EnEl_Dinero])
            Tabla_Regresion[:,2] = Caminos_Subyacente[:,t][Caminos_EnEl_Dinero]**3 #np.vectorize(mt.exp)(-Caminos_Subyacente[:,t][Caminos_EnEl_Dinero]/2)*(1-2*Caminos_Subyacente[:,t][Caminos_EnEl_Dinero]+(Caminos_Subyacente[:,t][Caminos_EnEl_Dinero]**2)/2)
            Modelo = LinearRegression().fit(Tabla_Regresion[:,0:3],Caminos_Derivado[:,t][Caminos_EnEl_Dinero])
            #print(Modelo.score(Tabla_Regresion[:,0:3],Caminos_Derivado[:,t][Caminos_EnEl_Dinero]))
            Tabla_Regresion[:,3] = Modelo.intercept_ + Modelo.coef_[0]*Tabla_Regresion[:,0] + Modelo.coef_[1]*Tabla_Regresion[:,1] + Modelo.coef_[2]*Tabla_Regresion[:,2] # Valor de Continuidad Esperado
            # Your next line is: Si E[HV]<EV entonces EV, HV En otro caso (OV)
            Caminos_Derivado[np.where(Caminos_EnEl_Dinero==True),t] = np.where(Tabla_Regresion[:,3]<(Caminos_Subyacente[:,t][Caminos_EnEl_Dinero]-Strike)*Call_Put,(Caminos_Subyacente[:,t][Caminos_EnEl_Dinero]-Strike)*Call_Put,Caminos_Derivado[:,t][Caminos_EnEl_Dinero])
            #Caminos_Derivado[np.where((Caminos_EnEl_Dinero==True)&(Tabla_Regresion[:,3]<(Caminos_Subyacente[:,t][Caminos_EnEl_Dinero]-Strike)*Call_Put)),t+1] = 0

    #return pd.DataFrame(Caminos_Subyacente)
    return Caminos_Derivado[:,0].mean()