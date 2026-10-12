import matplotlib.pyplot as plt
import numpy as np

# Data
datasets = ['Bushes', 'Outdoor Small\nPlant', 'Tree 1', 'Tree 5']
aligned_error = [20.69, 3.89, 36.29, 50.19]
postprocessed_error = [13.01, 28.23, 17.58, 1.28]

x = np.arange(len(datasets))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, aligned_error, width, label='Aligned 3DGS Error', color='indianred')
rects2 = ax.bar(x + width/2, postprocessed_error, width, label='Post-processed 3DGS Error', color='mediumseagreen')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Relative LAI Error (%)', fontsize=12)
ax.set_title('Effect of Floater Removal on LAI Estimation Error', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11)
ax.legend()

# Add values on top of bars
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.savefig('/home/intern/LAI/Report/lai_error_reduction.png', dpi=300)
print("Plot saved to lai_error_reduction.png")
