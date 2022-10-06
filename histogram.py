import matplotlib.pyplot as plt

a = [0.636, -0.03, 0.471, 0.837, 0.358]
b = [0.444, -0.28, 0.401, 0.842, -0.671]

plt.bar([i for i in range(len(a)*2)], a+b)
plt.ylim(-1,1)
plt.show()
