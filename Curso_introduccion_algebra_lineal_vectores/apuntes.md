# Curso de Introducción al Álgebra Lineal: Vectores

¿Qué veremos en este curso?

- Vectores
- Funciones lineales
- Norma y distancia
- Clustering

## Vectores

### Algunos objetos matemáticos: vectores y escalares

Vector columna = \
|1|\
|2|\
|3|

Vector renglón = [1,2,3]

Los apuntes de esta clase se pueden encontrar aquí: [apuntes](https://colab.research.google.com/drive/1bBQM1NGeyGFesOgCTs9bWGRu7UqDWeQ7?usp=sharing#scrollTo=1jAWFRqlXRRZ)

Un vector con "n" elementos tiene tamaño/longtiud de n.

Un número **SIN** tomar en cuenta que está dentro de un vector se llamará **escalar**. Si **está dentro** de un vector le llamaremos **coeficiente**.

Entonces un 3-vector tiene 3 coeficientes.

1.1 puede ser conficiente y también escalar.

#### Números reales

- Números enteros
- Números decimales

### Convención en notación

Los apuntes de esta clase se pueden encontrar aquí: [apuntes](https://colab.research.google.com/drive/1bBQM1NGeyGFesOgCTs9bWGRu7UqDWeQ7?usp=sharing#scrollTo=1jAWFRqlXRRZ)

### Comienza a utilizar vectores en Python

El modelo aditivo RGB

Los apuntes de esta clase se pueden encontrar aquí: [apuntes](https://colab.research.google.com/drive/1bBQM1NGeyGFesOgCTs9bWGRu7UqDWeQ7?usp=sharing#scrollTo=1jAWFRqlXRRZ)

- En python, los corchetes son LISTAS, son mutables
- En python, los paréntesis son TUPLAS, NO son mutables

La suma de listas en python regresa una concatenación, ejemplo:

```python
rojo = [255,0,0]
verde = [0,255,0]
Rojo_y_Verde = rojo+verde
print('El vector resultante de concatenar los vectores rojo y verde es',Rojo_y_Verde)
# El vector resultante de concatenar los vectores rojo y verde es [255, 0, 0, 0, 255, 0]
```

### Adición entre vectores

Los apuntes de esta clase se pueden encontrar aquí: [apuntes](https://colab.research.google.com/drive/1e5jw2bT7xaUSZRZCTSo8-WC2udYaHH32?usp=sharing)

Los vectores deben ser de la misma dimensión para poder ser sumados.

*conmutativa* : a + b = b + a

*asociativa* : (a+b) + c = a+(b+c) = a+b+c

### Suma de vectores en python

Para esto podemos usar el método zip de Python.
Documentación para este método aquí: [zip en python](https://www.programiz.com/python-programming/methods/built-in/zip)

Por ejemplo, tenemos:

```python
rojo = [255,0,0]
negro = [0,0,0]
def suma_vectores(a,b):
    return [i+j for i,j in zip(a,b)]


print('suma_vectores(rojo,negro) nos devuelve',suma_vectores(rojo,negro))
# resultado: [255,0,0]

print('suma_vectores(verde,negro) nos devuelve',suma_vectores([0,255,0],negro))
#imprime [0,255,0]
```

Sin embargo numpy es la libreria que se usa para manejar vectores y matrices en Python.

```python
import numpy as np #np es el estándar para el alias de numpy

rojo = np.array(rojo)
negro = np.array(negro)
print('La suma de los numpy array rojo + negro es', rojo+negro)
#Haciendo esto, ahora sí nos regresará la suma de los vectores
```

### Producto escalar-vector

Los apuntes de esta clase se pueden encontrar aquí: [apuntes](https://colab.research.google.com/drive/1e68pDA6HKsM0SpfXZmjxk2FO6_4mJdaY?usp=sharing)

**Recordemos que NO podemos sumar vectores de diferente dimensión.**

### Producto escalar-vector en Python

Los apuntes de esta clase se pueden encontrar aquí: [apuntes](https://colab.research.google.com/drive/1e68pDA6HKsM0SpfXZmjxk2FO6_4mJdaY?usp=sharing)

```python
a = np.linspace(0,5,1000)
# desde dónde, hasta donde, dividido en tantas partes iguales
```

### Producto interno

Los apuntes de esta clase se pueden encontrar aquí: [apuntes](https://colab.research.google.com/drive/1_Mt-whATHE1Xnal2X6LCF5QR5H1z56Am?usp=sharing)

Para recordar: **Producto punto o producto escalar nos regresa un número**

Para recordar: **La transpuesta de una columna nos dará un reglón y viceversa**

Para recordar: **Al multiplicar un vector renglón por un vector columna, tendremos un escalar.**

### Producto interno en Python

Podemos usar el "@" para indicar una multiplicación de vectores en Python.

Los apuntes de esta clase se pueden encontrar aquí: [apuntes](https://colab.research.google.com/drive/1_Mt-whATHE1Xnal2X6LCF5QR5H1z56Am?usp=sharing)

### Proyecto: análisis de sentimientos

```python
import numpy as np
import pandas as pd


def feeling(Tweet):
  tweet = Tweet.replace("!","").replace(",","").replace(".","").lower().split(" ")

  palabras =['muerte','pérdida','luto','excelente','gran','positivo','bueno','inteligente','ignorante','platzi','aprender','estudio','bien','quiero']

  palabras_positivas =["excelente","gran","quiero","positivo",'bien','positivo','bueno','inteligente']
  palabras_neutras = ["pérdida",'aprender','estudio','platzi']
  palabras_negativas = ["muerte","luto",'ignorante']

  w = []
  positivas = 0
  neutras = 0
  negativas = 0


  for i in palabras:
    w.append(tweet.count(i))
    if i in tweet and i in palabras_positivas:
      positivas += 1
    elif i in tweet and i in palabras_neutras:
      neutras += 1
    elif i in tweet and i in palabras_negativas:
      negativas += 1

  s = np.array([positivas,neutras,negativas])
  w = np.array(w)

  avg = (np.ones(w.size)/w.size).T.dot(w)
  score = s/(s[0]+s[1]+s[2])
  return Tweet,avg,score[0],score[1],score[2]

tweet1 = "Gran mexicano y excelente en su área, su muerte es una enorme perdida y debería ser luto nacional!!!"

tweet2 = "Vaya señora que bueno que se asesora por alguien inteligente no por el ignorante del Gatt"

tweet3 = "Se me ocurre y sin ver todos los videos de Plazti que me informéis por dónde empiezo. Entiendo que os tendría que decir quién soy y que quiero, vamos conocerme para asesorarme bien. Un saludo"

tweet4 = "Soy docente universitario, estoy intentando preparar mis clases en modo platzi bien didáctico, (le llamo modo noticiero), descargue una plataforma gratuita de grabación y transmisión de vídeo, se llama Obs estudio!bueno la sigo remando con sus funciones pero sé que saldrá algo!"

tweets = [tweet1,tweet2,tweet3,tweet4]
resultados = []

for j in tweets:
  resultados.append(feeling(j))

df = pd.DataFrame(resultados, columns=["Tweet","Calidad","P_positiva","P_neutra","P_negativa"])
df
```

## Funciones lineales

Las funciones van a tomar vectores y los convertirán en escalares.

Las funciones deben estar declaradas de manera explícita para poder trabajar con ellas.

**Funciones biyectivas** : 1 a 1

Los apuntes de esta clase se pueden encontrar aquí [apuntes](https://colab.research.google.com/drive/1uHLHnGyq5fgP917lwwh3JDNhNtr3EZhw?usp=sharing)

```python
import numpy as np

def f(x):
    return np.sum(x) # sumar todas las componentes del vector
    # función suma
```

```python
def g(x):
    return x[0] #regresa solamente la primera posición
    #proyección sobre x0
```

### Algunas funciones lineales

Reto 7:

```python
def g(x):
  return x[0]
g0 = g(np.array([0]))
g1 = g(np.array([0,0,1]))
g2 = g(np.array([1,0,4]))

alfa = 8 #numero random

if f(alfa * g0) == alfa * (f(g0)):
  print('g0 es homogenea')

if f(alfa * g1) == alfa * (f(g1)):
  print('g1 es homogenea')

if f(alfa * g2) == alfa * (f(g2)):
  print('g2 es homogenea')
```

Reto 7 pero mejor hecho:

```python
import numpy as np

def g(x):
  return x[0]


a = np.array([1,1,1,1])
b = np.array([1,0,1,0])
x,y = 1,-2 #valores random

if (g(np.dot(x,a) + np.dot(y,b))) == (np.dot(x,g(a)) + np.dot(y,g(b))):
  print("La función es homogenea")
```

### Un teorema en funciones lineales

Para consultar los apuntes de esta clase, [aquí están estos apuntes](https://colab.research.google.com/drive/1uHLHnGyq5fgP917lwwh3JDNhNtr3EZhw?usp=sharing)

La función de máximo **no** cumple la superposición, por lo tanto **no** es una función lineal.

Cuando queremos demostrar que algo no es verdad, basta con mostrar un simple ejemplo.

### Funciones afines

Para consultar los apuntes de esta clase, [aquí están estos apuntes](https://colab.research.google.com/drive/1uHLHnGyq5fgP917lwwh3JDNhNtr3EZhw?usp=sharing)

Reto 8:

```python
import numpy as np
P = np.array([[10,10,10,1] , [100,10,10,1] , [10,100,10,1] , [10,10,100,1]])
T = np.array([35,60,75,65])
x = np.linalg.solve(P,T)
print(f"La solución para el sistema de ecuaciones lineales es {x}")
```

### Aproximaciones de Taylor

Nuestra realidad es muy compleja. A veces nos conviene aproximar funciones lineales o afines.

Consultar los [apuntes](https://colab.research.google.com/drive/1JNNq5pH4Zx0QiGYE1jx6c2mhdrRVeETS?usp=sharing) para esta clase.

### Ejemplos de aproximaciones de Taylor

Reto 9:

```python
import numpy as np
def gradF(z):
  return np.array([1-np.exp(z[1]-z[0]) , np.exp(z[1]-z[0])])


def F(x):
  return x[0] + np.exp(x[1] - x[0])

print(F([1,2]))
print(F([2,4]))
print(F([3,6]))
print(F([4,8]))

def taylor_F(x,z):
  fz = F(z)
  gz = gradF(z)
  return fz + gz@(x-z)

print(taylor_F(np.array([1,2]) , np.array([1,2])))
print(taylor_F(np.array([2,4]) , np.array([1,2])))
print(taylor_F(np.array([3,6]) , np.array([1,2])))
print(taylor_F(np.array([4,8]) , np.array([1,2])))

error1 = np.abs(F([1,2] - taylor_F(np.array([1,2]) , np.array([1,2]))))
print(f'Error1: {error1}')

error2 = np.abs(F([1,2] - taylor_F(np.array([2,4]) , np.array([1,2]))))
print(f'Error2: {error2}')

error3 = np.abs(F([1,2] - taylor_F(np.array([3,6]) , np.array([1,2]))))
print(f'Error3: {error3}')

error4 = np.abs(F([1,2] - taylor_F(np.array([4,8]) , np.array([1,2]))))
print(f'Error4: {error4}')
```

### Un modelo de regresión

Apuntes de esta clase [están aquí](https://colab.research.google.com/drive/14cwjM298LQPc4h2O--gbdMOakyS7YCSE?usp=sharing)

Reto 10:

```python
import matplotlib.pyplot as plt # Para poder realizar visualizaciones
import pandas as pd # Para poder acceder a pandas
import numpy as np # Para poder tener vectores

df = pd.read_csv('/income_db_gorg.csv')

def pred(x):
    beta = np.array([6.5549034 + 2 , 5.75918372 + 2 , -2.94216316 - 1])
    v = 4152.02
    return x@beta + v

X = df[['Lat','Lon','Zip_Code']].values
Y_hat = pred(X)
Y = df['Mean'].values

fig, ax = plt.subplots(1,1,figsize=(7,7),dpi=120)

ax.scatter(Y_hat,Y,marker ='o',color='green')
ax.plot(Y,Y,ls='--')
plt.show()
```

## Norma y distancia

### Como calcular distancias de vectes

Para esta clase consultar [estos apuntes](https://colab.research.google.com/drive/1dGKOs5gpu1pE2tuMxOdENi3rNqRu39HT?usp=sharing)

Los vectores escalan, es decir, se hacen más grandes o se hacen más pequeños e incluso cambian de sentido.

Norma: longitud del vector.

$$ ||x|| $$

Así es como se expresa la norma.

Cuando sacamos raíz cuadrada, sabemos que podemos tener 2 soluciones, para en el caso de la norma siempre usaremos el valor positivo.

**Valor cuadrático medio** o *RMS*(Root Mean Square)

Consultar los apuntes de esta clase [aquí](https://colab.research.google.com/drive/1dGKOs5gpu1pE2tuMxOdENi3rNqRu39HT?usp=sharing)

### Distancia entre vectores y búsqueda de departamento

Apuntes de esta clase están [aquí](https://colab.research.google.com/drive/1dGKOs5gpu1pE2tuMxOdENi3rNqRu39HT?usp=sharing)

### Desviación estándar

Apuntes de esta clase están [aquí](https://colab.research.google.com/drive/1AWGDw7FhhyE1jHpZfJjyIzs3J_LASmIt?usp=sharing)

### Cálculo en el riesgo de inversiones

Apuntes de esta clase [aquí](https://colab.research.google.com/drive/1AWGDw7FhhyE1jHpZfJjyIzs3J_LASmIt?usp=sharing)

Reto 11:

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

a = np.array([0.1]*50)
b = np.sin(np.linspace(0,4,50)) + np.random.uniform(-0.1,0.1,50)
c = np.cos(np.linspace(2,6,50)) + np.random.uniform(-0.1,0.1,50)
d = [i - np.random.uniform(0,0.8) for i in np.linspace(0,1,50)]

az = stats.zscore(a)
bz = stats.zscore(b)
cz = stats.zscore(c)
dz = stats.zscore(d)

fig,ax  = plt.subplots(1,1,figsize=(15,7),dpi=120)
ax.plot(np.linspace(0,50,50),az, marker='o', linestyle='dashed',label='Inversión a')
ax.plot(np.linspace(0,50,50),bz, marker='o', linestyle='dashed',label='Inversión b')
ax.plot(np.linspace(0,50,50),cz, marker='o', linestyle='dashed',label='Inversión c')
ax.plot(np.linspace(0,50,50),dz, marker='o', linestyle='dashed',label='Inversión d')
ax.set_xlabel('Tiempo')
ax.set_ylabel('Fraccion de retorno')
ax.legend()
plt.show()

I = np.array([az,bz,cz,dz])
M = np.array([np.mean(x) for x in I])
S = np.array([np.std(x) for x in I])

fig,ax  = plt.subplots(1,1,figsize=(7,7),dpi=120)
ax.scatter(S[0],M[0], linestyle='dashed',label='Inversión a')
ax.scatter(S[1],M[1], linestyle='dashed',label='Inversión b')
ax.scatter(S[2],M[2], linestyle='dashed',label='Inversión c')
ax.scatter(S[3],M[3], linestyle='dashed',label='Inversión d')
ax.set_xlabel('Riesgo')
ax.set_ylabel('Retorno esperado')
ax.legend()
plt.show()
```

### Ángulo entre vectores y correlación

Apuntes de esta clase [aquí](https://colab.research.google.com/drive/1AWGDw7FhhyE1jHpZfJjyIzs3J_LASmIt?usp=sharing)

Cuando tenemos un ángulo de 90° los vectores son ortogonales y el producto punto es 0.
Se cumple la relación de:

$$ \cos(90) = 0 $$

Entender la correlación es fundamental para poder entender los fenómenos con series de tiempo.

Sin embargo, hay que recordar que ***correlación no significa causalidad.***
Puede que tengamos falsos positivos en las correlaciones aunque sí las comprobemos.

## Clustering

### ¿Qué es y por qué clustering?

Apuntes de esta clase [aquí](https://colab.research.google.com/drive/17v75OBB7DaeMjkpRoxG9n8N8xZnQXD5T?usp=sharing)

### Una aproximación: K-Means

Apuntes de esta clase [aquí](https://colab.research.google.com/drive/17v75OBB7DaeMjkpRoxG9n8N8xZnQXD5T?usp=sharing)

La convergencia en números reales es infinita. La aproximación de K-Means es subóptima, se acerca mucho al resultado que podría ser de manera exacta.

### K-Means en Python

Apuntes de esta clase [aquí](https://colab.research.google.com/drive/17v75OBB7DaeMjkpRoxG9n8N8xZnQXD5T?usp=sharing)

Algoritmo:

1. Aisgnamos dados los centroides aleatorios los puntos a esos clústers.
2. Vamos a actualizar los centroides en función de los puntos que fueron asignados a ese grupo.
