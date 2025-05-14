import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n = 1000
X1 = np.random.normal(loc=0, scale=0.95, size=n)
X2 = np.random.normal(loc=1, scale=1.05, size=n)

Q1_X1, Q3_X1 = np.percentile(X1, [25, 75])
Im_X1 = [Q1_X1, Q3_X1]
Out_X1 = [np.min(X1), np.max(X1)]

Q1_X2, Q3_X2 = np.percentile(X2, [25, 75])
Im_X2 = [Q1_X2, Q3_X2]
Out_X2 = [np.min(X2), np.max(X2)]

a_values = np.arange(0, 2, 0.01)
J_Im = []
J_Out = []

for a in a_values:
    shifted_Im_X1 = [Im_X1[0] + a, Im_X1[1] + a]
    shifted_Out_X1 = [Out_X1[0] + a, Out_X1[1] + a]
    
    intersection_low = max(shifted_Im_X1[0], Im_X2[0])
    intersection_high = min(shifted_Im_X1[1], Im_X2[1])
    if intersection_high > intersection_low:
        intersection = intersection_high - intersection_low
        union_low = min(shifted_Im_X1[0], Im_X2[0])
        union_high = max(shifted_Im_X1[1], Im_X2[1])
        union = union_high - union_low
        J_Im.append(intersection / union)
    else:
        J_Im.append(0)
    
    intersection_low = max(shifted_Out_X1[0], Out_X2[0])
    intersection_high = min(shifted_Out_X1[1], Out_X2[1])
    if intersection_high > intersection_low:
        intersection = intersection_high - intersection_low
        union_low = min(shifted_Out_X1[0], Out_X2[0])
        union_high = max(shifted_Out_X1[1], Out_X2[1])
        union = union_high - union_low
        J_Out.append(intersection / union)
    else:
        J_Out.append(0)

a_Im = a_values[np.argmax(J_Im)]
a_Out = a_values[np.argmax(J_Out)]

plt.figure(figsize=(10, 6))
plt.plot(a_values, J_Im, label='J_Inn')
plt.plot(a_values, J_Out, label='J_Out')
plt.xlabel('a')
plt.ylabel('J')
plt.legend()
plt.title('Графики J_Inn(a) и J_Out(a)')
plt.grid(True)
plt.show()

print(f"Оптимальный a для внутренних оценок: {a_Im:.2f}")
print(f"Оптимальный a для внешних оценок: {a_Out:.2f}")
