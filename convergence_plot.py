import matplotlib.pyplot as plt
import numpy as np

# Gradient Descent Data
iterations = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999]
theta0 = [2.036000, 2.597810, 2.801336, 2.901869, 2.951528, 2.976057, 2.988173, 2.994158, 2.997114, 2.998575, 2.999291]
theta1 = [1.019667, 2.055163, 2.027248, 2.013459, 2.006648, 2.003284, 2.001622, 2.000801, 2.000396, 2.000195, 2.000097]
cost = [14.980000, 0.058736, 0.014331, 0.003497, 0.000853, 0.000208, 0.000051, 0.000012, 0.000003, 0.000001, 0.000000]

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Gradient Descent Convergence Analysis', fontsize=14, fontweight='bold')

# Plot 1: Cost Function Convergence
axes[0, 0].plot(iterations, cost, 'b-o', linewidth=2, markersize=6)
axes[0, 0].set_xlabel('Iteration', fontsize=11)
axes[0, 0].set_ylabel('Cost (J)', fontsize=11)
axes[0, 0].set_title('Cost Function vs Iterations', fontsize=12)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xlim(-50, 1050)

# Plot 2: Cost Function (Log Scale) - Better visualization of convergence
axes[0, 1].semilogy(iterations, [c + 1e-10 for c in cost], 'r-o', linewidth=2, markersize=6)
axes[0, 1].set_xlabel('Iteration', fontsize=11)
axes[0, 1].set_ylabel('Cost (J) - Log Scale', fontsize=11)
axes[0, 1].set_title('Cost Function (Log Scale)', fontsize=12)
axes[0, 1].grid(True, alpha=0.3, which='both')
axes[0, 1].set_xlim(-50, 1050)

# Plot 3: Theta Parameters Convergence
axes[1, 0].plot(iterations, theta0, 'g-o', linewidth=2, markersize=6, label='θ₀ (intercept)')
axes[1, 0].plot(iterations, theta1, 'm-s', linewidth=2, markersize=6, label='θ₁ (slope)')
axes[1, 0].axhline(y=3, color='g', linestyle='--', alpha=0.5, label='θ₀ target ≈ 3')
axes[1, 0].axhline(y=2, color='m', linestyle='--', alpha=0.5, label='θ₁ target ≈ 2')
axes[1, 0].set_xlabel('Iteration', fontsize=11)
axes[1, 0].set_ylabel('Parameter Value', fontsize=11)
axes[1, 0].set_title('Parameter Convergence (θ₀ and θ₁)', fontsize=12)
axes[1, 0].legend(loc='right', fontsize=9)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xlim(-50, 1050)

# Plot 4: 3D-like visualization - Parameter Space Trajectory
axes[1, 1].scatter(theta0, theta1, c=iterations, cmap='viridis', s=100, edgecolors='black', linewidth=1)
axes[1, 1].plot(theta0, theta1, 'k--', alpha=0.5, linewidth=1)
axes[1, 1].scatter([3], [2], c='red', s=200, marker='*', edgecolors='black', linewidth=1, label='Optimal (3, 2)', zorder=5)
axes[1, 1].set_xlabel('θ₀ (intercept)', fontsize=11)
axes[1, 1].set_ylabel('θ₁ (slope)', fontsize=11)
axes[1, 1].set_title('Parameter Space Trajectory', fontsize=12)
axes[1, 1].legend(loc='lower right', fontsize=9)
axes[1, 1].grid(True, alpha=0.3)

# Add colorbar for iteration number
cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
cbar.set_label('Iteration', fontsize=10)

# Annotate start and end points
axes[1, 1].annotate('Start', xy=(theta0[0], theta1[0]), xytext=(theta0[0]-0.2, theta1[0]+0.15),
                    fontsize=9, ha='center', arrowprops=dict(arrowstyle='->', color='gray'))
axes[1, 1].annotate('End', xy=(theta0[-1], theta1[-1]), xytext=(theta0[-1]+0.15, theta1[-1]+0.1),
                    fontsize=9, ha='center')

plt.tight_layout()
plt.savefig('gradient_descent_convergence.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✓ Plot saved as 'gradient_descent_convergence.png'")
print("\nConvergence Summary:")
print(f"  Initial Cost: {cost[0]:.6f}")
print(f"  Final Cost:   {cost[-1]:.6f}")
print(f"  Cost Reduction: {((cost[0] - cost[-1]) / cost[0] * 100):.4f}%")
print(f"\n  θ₀: {theta0[0]:.6f} → {theta0[-1]:.6f} (target ≈ 3)")
print(f"  θ₁: {theta1[0]:.6f} → {theta1[-1]:.6f} (target ≈ 2)")
