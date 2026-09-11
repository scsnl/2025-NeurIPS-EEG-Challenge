import numpy as np

X = np.random.normal(0.023, 0.17, size = (2000000,))

P0 = np.round(np.mean(X**2)/np.std(X)**2,16)
print(np.mean(X**2)/np.std(X)**2, P0)
a = 0.012
Pa =  np.round(np.mean((X-a)**2)/np.std(X)**2,16)
print(np.mean((X-a)**2)/np.std(X)**2, Pa)
Pa2 = np.round(np.mean((X-a/2)**2)/np.std(X)**2,16)
print(np.mean((X-a/2)**2)/np.std(X)**2, Pa2)

#a = 0.024669726281615182
#P0 = 1.0093400478363037**2
#Pa = 1.0151599645614624**2 
#Pa2 = TBD**2

A = -4*Pa2/a**2
B = 2*Pa/a**2
C = 2*P0/a**2

E = (a*A+a*B/2+3*a*C/2)/(2*(A+B+C))
S = np.sqrt(1/(A+B+C))

print('Predicted mean:', E)
print('Predicted std:', S)

print('Empirical mean:', np.mean(X))
print('Empirical std:', np.std(X))