import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

def compare_models(runs_dir="runs/train"):
    """
    Scans the runs directory, extracts metrics from results.csv for all trained models,
    displays a summary table, and plots the comparison.
    """
    print(f"Scanning for model results in {runs_dir}...")
    
    # Find all results.csv files
    result_files = glob.glob(os.path.join(runs_dir, "crowd_model*", "results.csv"))
    
    if not result_files:
        print(f"No results found in {runs_dir}. Please ensure models have been trained.")
        return
    
    # We will try to infer the model type based on directory creation time or we just label them Model 1, 2, 3
    # Actually, we can check the arguments.yaml to find the exact model name, or just sort them
    
    model_data = []
    
    for file in sorted(result_files, key=os.path.getmtime):
        dir_name = os.path.basename(os.path.dirname(file))
        
        try:
            # Read the CSV. The column names have leading spaces.
            df = pd.read_csv(file)
            df.columns = df.columns.str.strip()
            
            # Extract final epoch metrics
            final_epoch = df.iloc[-1]
            precision = final_epoch.get('metrics/precision(B)', 0)
            recall = final_epoch.get('metrics/recall(B)', 0)
            map50 = final_epoch.get('metrics/mAP50(B)', 0)
            map50_95 = final_epoch.get('metrics/mAP50-95(B)', 0)
            
            model_data.append({
                'Run': dir_name,
                'Precision': precision,
                'Recall': recall,
                'mAP50': map50,
                'mAP50-95': map50_95
            })
        except Exception as e:
            print(f"Error processing {file}: {e}")
            
    if not model_data:
        print("No valid data could be extracted.")
        return
        
    # Create DataFrame
    results_df = pd.DataFrame(model_data)
    
    # Print Table
    print("\n" + "="*60)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*60)
    print(results_df.to_string(index=False))
    print("="*60 + "\n")
    
    # Identify Best Model
    best_model = results_df.loc[results_df['mAP50'].idxmax()]
    print(f"⭐ BEST MODEL OVERALL (based on mAP50): {best_model['Run']} with mAP50: {best_model['mAP50']:.4f}\n")
    
    # Plotting
    try:
        # We rename the 'Run' column to something simpler if it's too long
        labels = results_df['Run'].tolist()
        precision = results_df['Precision'].tolist()
        recall = results_df['Recall'].tolist()
        map50 = results_df['mAP50'].tolist()

        x = range(len(labels))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 6))

        # Bar charts
        ax.bar([i - width for i in x], precision, width, label='Precision', color='skyblue')
        ax.bar(x, recall, width, label='Recall', color='lightgreen')
        ax.bar([i + width for i in x], map50, width, label='mAP50', color='salmon')

        ax.set_ylabel('Scores')
        ax.set_title('YOLO Models Comparative Performance (UCF-QNRF)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.set_ylim([0, 1.1])

        # Add values on top of bars
        for i in x:
            ax.text(i - width, precision[i] + 0.02, f'{precision[i]:.2f}', ha='center', va='bottom', fontsize=9)
            ax.text(i, recall[i] + 0.02, f'{recall[i]:.2f}', ha='center', va='bottom', fontsize=9)
            ax.text(i + width, map50[i] + 0.02, f'{map50[i]:.2f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plot_path = "model_comparison_chart.png"
        plt.savefig(plot_path)
        print(f"Comparison chart saved as: {os.path.abspath(plot_path)}")
        
    except Exception as e:
        print(f"Could not generate plot: {e}")

if __name__ == "__main__":
    compare_models()
