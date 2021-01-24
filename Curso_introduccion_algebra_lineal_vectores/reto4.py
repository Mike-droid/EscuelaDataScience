import numpy as np

rojo_vector = [255,0,0]
verde_vector = [0,255,0]
a_vector = [0,0,255]
negro_vector = [0,0,0]

rojo = np.array(rojo_vector)
verde = np.array(verde_vector)
a = np.array(a_vector)
negro = np.array(negro_vector)

print(f'La suma del vector rojo y verde y a es {rojo+verde+a}')
print(f'La suma del vector rojo y el vector verdes es {rojo+verde}')
print(f'La resta de negro con a es {negro-a}')