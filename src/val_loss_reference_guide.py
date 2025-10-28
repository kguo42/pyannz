import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans' 

# Create synthetic loss curves for different scenarios
def load_guide():
    epochs = np.arange(500)

    # 1. IDEAL: Good generalization
    train_ideal = 1.0 * np.exp(-epochs/100) + 0.05
    val_ideal = 1.0 * np.exp(-epochs/100) + 0.08

    # 2. SLIGHT OVERFITTING: Acceptable
    train_slight = 1.0 * np.exp(-epochs/80) + 0.02
    val_slight = 1.0 * np.exp(-epochs/120) + 0.10

    # 3. MODERATE OVERFITTING: Warning signs
    train_moderate = 1.0 * np.exp(-epochs/60) + 0.01
    val_moderate = 1.0 * np.exp(-epochs/150) + 0.15
    val_moderate[200:] = val_moderate[200:] + np.linspace(0, 0.05, len(val_moderate[200:]))

    # 4. SEVERE OVERFITTING: Bad
    train_severe = 1.0 * np.exp(-epochs/50) + 0.005
    val_severe = 1.0 * np.exp(-epochs/100) + 0.10
    val_severe[150:] = val_severe[150] + np.linspace(0, 0.3, len(val_severe[150:]))

    # 5. UNDERFITTING: Model too simple
    train_underfit = 0.5 * np.exp(-epochs/200) + 0.3
    val_underfit = 0.5 * np.exp(-epochs/200) + 0.32

    # 6. UNSTABLE TRAINING: Learning rate too high
    np.random.seed(42)
    train_unstable = 0.5 * np.exp(-epochs/80) + 0.1 + 0.05 * np.random.randn(len(epochs))
    val_unstable = 0.5 * np.exp(-epochs/80) + 0.15 + 0.08 * np.random.randn(len(epochs))
    train_unstable = np.maximum(train_unstable, 0.05)
    val_unstable = np.maximum(val_unstable, 0.05)

    # Create the plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Validation Loss Patterns: What They Mean', fontsize=16, fontweight='bold')

    # 1. Ideal
    ax = axes[0, 0]
    ax.plot(epochs, train_ideal, label='Train Loss', linewidth=2, color='blue')
    ax.plot(epochs, val_ideal, label='Val Loss', linewidth=2, color='red')
    ax.set_title('IDEAL: Good Generalization', fontsize=12, fontweight='bold', color='green')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(250, 0.5, 'Val loss slightly higher\nbut stable.\nSmall gap = good!',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    # 2. Slight overfitting
    ax = axes[0, 1]
    ax.plot(epochs, train_slight, label='Train Loss', linewidth=2, color='blue')
    ax.plot(epochs, val_slight, label='Val Loss', linewidth=2, color='orange')
    ax.set_title(' ACCEPTABLE: Slight Overfitting', fontsize=12, fontweight='bold', color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(250, 0.5, 'Val loss ~2-3x train.\nStill converging.\nOK for most cases.',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    # 3. Moderate overfitting
    ax = axes[0, 2]
    ax.plot(epochs, train_moderate, label='Train Loss', linewidth=2, color='blue')
    ax.plot(epochs, val_moderate, label='Val Loss', linewidth=2, color='darkorange')
    best_epoch = np.argmin(val_moderate)
    ax.axvline(x=best_epoch, color='green', linestyle='--', linewidth=2, label=f'Stop here (epoch {best_epoch})')
    ax.set_title('MODERATE: Early Stopping Needed', fontsize=12, fontweight='bold', color='darkorange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(300, 0.3, 'Val loss increases\nafter epoch ~200.\nUse early stopping!',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # 4. Severe overfitting
    ax = axes[1, 0]
    ax.plot(epochs, train_severe, label='Train Loss', linewidth=2, color='blue')
    ax.plot(epochs, val_severe, label='Val Loss', linewidth=2, color='red')
    ax.set_title('SEVERE: Bad Overfitting', fontsize=12, fontweight='bold', color='darkred')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(250, 0.3, 'Val loss diverges!\nModel too complex.\nReduce capacity.',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

    # 5. Underfitting
    ax = axes[1, 1]
    ax.plot(epochs, train_underfit, label='Train Loss', linewidth=2, color='blue')
    ax.plot(epochs, val_underfit, label='Val Loss', linewidth=2, color='purple')
    ax.set_title(' UNDERFITTING: Model Too Simple', fontsize=12, fontweight='bold', color='purple')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(250, 0.4, 'Both losses high.\nNot learning well.\nIncrease capacity.',
            bbox=dict(boxstyle='round', facecolor='plum', alpha=0.7))

    # 6. Unstable
    ax = axes[1, 2]
    ax.plot(epochs, train_unstable, label='Train Loss', linewidth=2, color='blue', alpha=0.7)
    ax.plot(epochs, val_unstable, label='Val Loss', linewidth=2, color='brown', alpha=0.7)
    ax.set_title(' UNSTABLE: Learning Rate Issues', fontsize=12, fontweight='bold', color='brown')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(250, 0.4, 'Wild oscillations.\nReduce learning rate\nor check data.',
            bbox=dict(boxstyle='round', facecolor='bisque', alpha=0.7))

    plt.tight_layout()
    plt.show()

    # Summary table
    print("=" * 70)
    print("VALIDATION LOSS PATTERNS - QUICK REFERENCE")
    print("=" * 70)
    print("\n✅ IDEAL:")
    print("   - Val loss slightly > train loss (1.2-1.5x)")
    print("   - Both decrease smoothly")
    print("   - Val loss plateaus at similar level to train")
    print("   - Small gap remains constant")

    print("\n⚠️ ACCEPTABLE (Slight Overfitting):")
    print("   - Val loss 2-3x train loss")
    print("   - Both still decreasing")
    print("   - Gap widens slowly")
    print("   - Solution: Train a bit longer, add slight regularization")

    print("\n⚠️ MODERATE (Early Stopping Needed):")
    print("   - Val loss decreases then increases")
    print("   - Train loss continues to decrease")
    print("   - Clear divergence point visible")
    print("   - Solution: Use early stopping at minimum val loss")

    print("\n❌ SEVERE (Bad Overfitting):")
    print("   - Val loss >> train loss (5-30x or more)")
    print("   - Val loss increases significantly over time")
    print("   - Train loss very low")
    print("   - Solution: Reduce model size, add dropout, stronger regularization")

    print("\n❌ UNDERFITTING:")
    print("   - Both losses high and similar")
    print("   - Little to no gap")
    print("   - Both plateau at high values")
    print("   - Solution: Increase model capacity, train longer, reduce regularization")

    print("\n❌ UNSTABLE:")
    print("   - Wild oscillations in both losses")
    print("   - No smooth convergence")
    print("   - Large spikes")
    print("   - Solution: Reduce learning rate, check data for issues, use gradient clipping")
    print("=" * 70)