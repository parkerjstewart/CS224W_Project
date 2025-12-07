import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

PLAYERS = "../data/player_features.csv"

def comparative_eval(filename):
    """
    Implements different eval technique comparing
    probability of one direction vs the other, rather than
    probability of each direction vs a threshold of 0.5.

    All labes are 1 in the prediction files.
    
    Args:
        filename: Path to the CSV file
        
    Returns:
        Percentage of rows where pred > reverse_pred
    """
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        total = 0
        count = 0
        
        for row in reader:
            pred = float(row['pred'])
            reverse_pred = float(row['reverse_pred'])
            
            if (pred > reverse_pred):
                count += 1
            total += 1
    
    return (count / total * 100) if total > 0 else 0


def traditional_two_way_eval(filename):
    """
    Calculates success of row twice, once for if forward pred is correct 
    and once for if reverse pred is correct in the traditional eval technique.

    All labels are 1 in the prediction files.
    
    Args:
        filename: Path to the CSV file
        
    Returns:
        Percentage of correct rows
    """
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        total = 0
        count = 0
        
        for row in reader:
            pred = float(row['pred'])
            rev_pred = float(row['reverse_pred'])
            label = int(row['label'])
            
            if pred > 0.5 and label == 1:
                count += 1
            if rev_pred < 0.5 and label == 1:
                count += 1
            total += 2
    
    return (count / total * 100) if total > 0 else 0

def plot_confidence_by_rank_difference(filename):
    """
    Plot scatter graph with line of best fit:
    - pred vs (p2_rank - p1_rank)
    - reverse_pred vs (p1_rank - p2_rank)
    
    Args:
        predictions_file: Path to predictions CSV
        players_file: Path to players CSV
    """
    # Load data
    preds = pd.read_csv(filename)
    players = pd.read_csv(PLAYERS)
    
    # Create a dictionary for quick rank lookup
    rank_dict = dict(zip(players['player_id'], players['current_rank']))
    
    # Get ranks for p1 and p2
    preds['p1_rank'] = preds['p1_id'].map(rank_dict)
    preds['p2_rank'] = preds['p2_id'].map(rank_dict)
    
    # Calculate rank differences
    preds['p2_minus_p1'] = preds['p2_rank'] - preds['p1_rank']
    preds['p1_minus_p2'] = preds['p1_rank'] - preds['p2_rank']
    
    # Remove rows with missing ranks
    preds = preds.dropna(subset=['p1_rank', 'p2_rank'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot pred vs (p2_rank - p1_rank)
    ax.scatter(preds['p2_minus_p1'], preds['pred'], 
               alpha=0.5, label='pred vs (p2_rank - p1_rank)', 
               color='blue', s=20)
    
    # Plot reverse_pred vs (p1_rank - p2_rank)
    ax.scatter(preds['p1_minus_p2'], preds['reverse_pred'], 
               alpha=0.5, label='reverse_pred vs (p1_rank - p2_rank)', 
               color='red', s=20)
    
    # Calculate and plot line of best fit for pred
    z1 = np.polyfit(preds['p2_minus_p1'], preds['pred'], 1)
    p1 = np.poly1d(z1)
    x_line = np.linspace(preds['p2_minus_p1'].min(), preds['p2_minus_p1'].max(), 100)
    ax.plot(x_line, p1(x_line), 'b-', linewidth=2, 
            label=f'Best fit (pred): y={z1[0]:.4f}x+{z1[1]:.4f}')
    
    # Calculate and plot line of best fit for reverse_pred
    z2 = np.polyfit(preds['p1_minus_p2'], preds['reverse_pred'], 1)
    p2 = np.poly1d(z2)
    x_line2 = np.linspace(preds['p1_minus_p2'].min(), preds['p1_minus_p2'].max(), 100)
    ax.plot(x_line2, p2(x_line2), 'r-', linewidth=2, 
            label=f'Best fit (reverse_pred): y={z2[0]:.4f}x+{z2[1]:.4f}')
    
    ax.set_xlabel('Rank Difference', fontsize=12)
    ax.set_ylabel('Prediction Value', fontsize=12)
    ax.set_title('Prediction vs Rank Difference', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# Example usage:
if __name__ == "__main__":

    for filename in os.listdir("../predictions"):
        filename = os.path.join("../predictions", filename)
        print("="*50)
        print(f"Analyzing file: {filename}")
        print("="*50, "\n")

        pct_pred_greater = comparative_eval(filename)
        print(f"Comparative eval: {pct_pred_greater:.2f}%")
        
        pct_above_threshold = traditional_two_way_eval(filename)
        print(f"Traditional two-way eval: {pct_above_threshold:.2f}%")

        fig = plot_confidence_by_rank_difference(filename)
        plt.show()