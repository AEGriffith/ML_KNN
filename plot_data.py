import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

def hist(f0, f1, f2, f3, f4, f5, f6, share = True):
    fig, (ax0, ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows=7, figsize=(8, 6), sharex=share, sharey=share)
    ax0.hist(f0)
    ax1.hist(f1)
    ax2.hist(f2)
    ax3.hist(f3)
    ax4.hist(f4)
    ax5.hist(f5)
    ax6.hist(f6)
    plt.show()