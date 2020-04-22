from matplotlib import pyplot as plt

def plot(x1,x2,title,cc):
    plt.scatter(x1,x2,c=cc)
    plt.title(title)
    plt.xlabel("X-coordinates")
    plt.ylabel("Y-coordinates")
    plt.show()
