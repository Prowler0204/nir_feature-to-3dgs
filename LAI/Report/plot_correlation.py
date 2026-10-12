import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

# Data
datasets = ['Bushes', 'Outdoor Small Plant', 'Tree 1', 'Tree 5']
tls_lai = np.array([0.5656, 0.4091, 0.1422, 0.1333])
postprocessed_lai = np.array([0.6392, 0.5246, 0.1172, 0.1316])

# Calculate Pearson Correlation and R-squared
pearson_r, _ = pearsonr(tls_lai, postprocessed_lai)
r_squared = pearson_r ** 2

# Plotting
plt.figure(figsize=(8, 6))

# Scatter points
plt.scatter(tls_lai, postprocessed_lai, color='blue', s=100, label='Data Points', zorder=5)

# Add labels to points
for i, txt in enumerate(datasets):
    plt.annotate(txt, (tls_lai[i], postprocessed_lai[i]), textcoords="offset points", xytext=(0,10), ha='center')

# Linear regression line (best fit)
m, b = np.polyfit(tls_lai, postprocessed_lai, 1)
plt.plot(tls_lai, m*tls_lai + b, color='red', linestyle='--', label=f'Best Fit Line (R² = {r_squared:.2f})')

# 1:1 Reference Line (Perfect Agreement)
min_val = min(min(tls_lai), min(postprocessed_lai)) - 0.05
max_val = max(max(tls_lai), max(postprocessed_lai)) + 0.05
plt.plot([min_val, max_val], [min_val, max_val], color='green', linestyle=':', label='1:1 Reference Line')

# Formatting
plt.title(f'Correlation between TLS LAI and Post-processed 3DGS LAI\nPearson R = {pearson_r:.3f}', fontsize=14)
plt.xlabel('TLS LAI (Reference)', fontsize=12)
plt.ylabel('3DGS LAI (Estimated)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

# Save the plot
output_path = '/home/intern/LAI/Report/correlation_plot.png'
plt.savefig(output_path, dpi=300)
print(f"Plot saved successfully to {output_path}")
print(f"Pearson R: {pearson_r:.3f}")
