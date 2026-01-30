"""
Script to plot length vs average faithfulness score from evaluation JSONL file.
"""
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
    
    # Try to configure LaTeX rendering
    try:
        # Test if LaTeX is available
        from matplotlib import mathtext
        rcParams['text.usetex'] = True
        rcParams['font.family'] = 'serif'
        rcParams['font.serif'] = ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif']
        USE_LATEX = True
    except Exception:
        # Fall back to regular text with math rendering
        rcParams['text.usetex'] = False
        rcParams['font.family'] = 'serif'
        rcParams['mathtext.fontset'] = 'cm'  # Computer Modern for math
        USE_LATEX = False
    
    # Configure font sizes
    rcParams['font.size'] = 11
    rcParams['axes.labelsize'] = 12
    rcParams['axes.titlesize'] = 14
    rcParams['xtick.labelsize'] = 10
    rcParams['ytick.labelsize'] = 10
    rcParams['legend.fontsize'] = 10
    rcParams['figure.titlesize'] = 16
    
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    USE_LATEX = False
    print("Error: matplotlib not available. Install with: pip install matplotlib")
    sys.exit(1)


def load_jsonl_data(file_path: str) -> List[Dict]:
    """
    Load data from JSONL file.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of dictionaries containing length and score
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    if 'length' in entry and 'score' in entry:
                        data.append(entry)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
                    continue
    return data


def calculate_average_scores_by_length(data: List[Dict]) -> Dict[int, Dict]:
    """
    Calculate average scores grouped by length.
    
    Args:
        data: List of dictionaries with 'length' and 'score' keys
        
    Returns:
        Dictionary mapping length to statistics (mean, min, max, count)
    """
    scores_by_length = defaultdict(list)
    
    for entry in data:
        length = entry['length']
        score = entry['score']
        # Only include valid scores (not None)
        if score is not None:
            scores_by_length[length].append(score)
    
    # Calculate statistics for each length
    stats_by_length = {}
    for length in sorted(scores_by_length.keys()):
        scores = scores_by_length[length]
        stats_by_length[length] = {
            "mean": sum(scores) / len(scores),
            "min": min(scores),
            "max": max(scores),
            "count": len(scores),
        }
    
    return stats_by_length


def plot_length_vs_score(
    stats_by_length: Dict[int, Dict],
    output_path: str = None,
    title: str = "Average Faithfulness Score vs Traversal Length"
):
    """
    Create a styled plot of length vs average score with LaTeX rendering.
    
    Args:
        stats_by_length: Dictionary mapping length to statistics
        output_path: Path to save the plot (if None, displays)
        title: Plot title
    """
    if not stats_by_length:
        print("No data to plot.")
        return
    
    lengths = sorted(stats_by_length.keys())
    avg_scores = [stats_by_length[length]["mean"] for length in lengths]
    counts = [stats_by_length[length]["count"] for length in lengths]
    min_scores = [stats_by_length[length]["min"] for length in lengths]
    max_scores = [stats_by_length[length]["max"] for length in lengths]
    
    # Create figure with better styling
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#fafafa')
    
    # Color scheme
    primary_color = '#2E86AB'  # Professional blue
    accent_color = '#A23B72'  # Purple accent
    grid_color = '#e0e0e0'
    
    # Plot average scores with gradient color
    line = ax.plot(lengths, avg_scores, marker='o', linestyle='-', linewidth=2.5, 
                   markersize=10, label=r'$\bar{s}$ (Average Score)', 
                   color=primary_color, markerfacecolor=primary_color,
                   markeredgecolor='white', markeredgewidth=1.5, zorder=3)
    
    # Add shaded error region (min to max)
    ax.fill_between(lengths, min_scores, max_scores, alpha=0.2, 
                    color=primary_color, label=r'Range (min--max)', zorder=1)
    
    # Add count annotations with LaTeX
    for i, (length, avg_score, count) in enumerate(zip(lengths, avg_scores, counts)):
        ax.annotate(
            r'$n=' + str(count) + r'$',
            xy=(length, avg_score),
            xytext=(8, 8),
            textcoords='offset points',
            fontsize=9,
            alpha=0.8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor=primary_color, alpha=0.7, linewidth=0.5)
        )
    
    # Labels with LaTeX
    ax.set_xlabel(r'Traversal Length $\ell$ (Number of Formulas)', 
                  fontsize=13, fontweight='bold', color='#333333')
    ax.set_ylabel(r'Average Faithfulness Score $\bar{s}$', 
                  fontsize=13, fontweight='bold', color='#333333')
    
    # Title with LaTeX (if available) or regular text
    try:
        use_latex = rcParams.get('text.usetex', False)
        if use_latex:
            ax.set_title(r'\textbf{' + title.replace(' vs ', r' vs ') + '}', 
                        fontsize=15, pad=15, color='#1a1a1a')
        else:
            ax.set_title(title, fontsize=15, fontweight='bold', pad=15, color='#1a1a1a')
    except Exception:
        # Fallback if LaTeX rendering fails
        ax.set_title(title, fontsize=15, fontweight='bold', pad=15, color='#1a1a1a')
    
    # Styled grid
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8, color=grid_color, zorder=0)
    ax.set_axisbelow(True)
    
    # Set limits and ticks
    ax.set_ylim(0, 10)
    ax.set_xlim(min(lengths) - 0.5, max(lengths) + 0.5)
    ax.set_xticks(lengths)
    ax.set_yticks(np.arange(0, 11, 1))
    
    # Add horizontal reference line at y=5
    ax.axhline(y=5, color=accent_color, linestyle='--', linewidth=1.5, 
               alpha=0.6, label=r'Midpoint ($s = 5.0$)', zorder=2)
    
    # Styled legend
    legend = ax.legend(loc='best', frameon=True, fancybox=True, 
                      shadow=True, framealpha=0.95, edgecolor='#cccccc')
    legend.get_frame().set_facecolor('white')
    
    # Style the spines
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')
        spine.set_linewidth(1)
    
    # Add subtle border
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def print_statistics(stats_by_length: Dict[int, Dict]):
    """Print statistics for each length."""
    print("\n" + "="*80)
    print("Statistics by Traversal Length:")
    print("="*80)
    print(f"{'Length':<10} {'Mean':<10} {'Min':<10} {'Max':<10} {'Count':<10}")
    print("-"*80)
    
    for length in sorted(stats_by_length.keys()):
        stats = stats_by_length[length]
        print(f"{length:<10} {stats['mean']:<10.2f} {stats['min']:<10.2f} "
              f"{stats['max']:<10.2f} {stats['count']:<10}")
    
    print("="*80)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Plot length vs average faithfulness score from JSONL file"
    )
    parser.add_argument(
        "jsonl_file",
        type=str,
        help="Path to JSONL file containing evaluation results"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the plot (default: same directory as input with .png extension)"
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Average Faithfulness Score vs Traversal Length",
        help="Plot title"
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.jsonl_file):
        print(f"Error: File not found: {args.jsonl_file}")
        sys.exit(1)
    
    # Load data
    print(f"Loading data from: {args.jsonl_file}")
    data = load_jsonl_data(args.jsonl_file)
    
    if not data:
        print("Error: No valid data found in file.")
        sys.exit(1)
    
    print(f"Loaded {len(data)} entries")
    
    # Calculate statistics
    stats_by_length = calculate_average_scores_by_length(data)
    
    if not stats_by_length:
        print("Error: No valid scores found in data.")
        sys.exit(1)
    
    # Print statistics
    print_statistics(stats_by_length)
    
    # Determine output path
    if args.output is None:
        base_name = os.path.splitext(args.jsonl_file)[0]
        output_path = f"{base_name}_plot.png"
    else:
        output_path = args.output
    
    # Create plot
    print(f"\nGenerating plot...")
    plot_length_vs_score(stats_by_length, output_path, args.title)
    
    print(f"\nDone!")


if __name__ == "__main__":
    main()
