import matplotlib.pyplot as plt

filename = "C:\\Users\\Jalynn\\Documents\\NAS\\中期报告\\Global_NAS_中期数据\\Global_NAS_中期数据\\Global_NAS\\25.txt"
sc = []
with open(filename, "rb") as f:
    for line in f:
        if float(line.split()[0]) > 0.2:
            sc.append(float(line.split()[0]))

x = [i for i in range(len(sc))]
font = {
    'size': 30,
}
plt.figure(figsize=(10, 9))
plt.plot(x, sc, color='red', lw=3)
plt.tick_params(labelsize=30)
plt.xlabel('Sample Number', font)
plt.ylabel('Score', font)
plt.savefig('C:\\Users\\Jalynn\\Desktop\\macro-25.pdf')
plt.show()
