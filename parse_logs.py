import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def parse_log(log_file):
    val_accs = []
    test_accs = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if line.startswith('accuracy:'):
                acc = float(line.split('accuracy:')[1].strip())
                test_accs.append(acc)
            elif line.startswith('val_accuracy:'):
                acc = float(line.split('val_accuracy:')[1].strip())
                val_accs.append(acc)
            elif line.startswith('test_accuracy:'):  
                acc = float(line.split('test_accuracy:')[1].strip())
                test_accs.append(acc)

    return val_accs, test_accs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='logs/classification')
    parser.add_argument("--output_file", type=str, default='results.csv')
    parser.add_argument("--agg", action='store_true', help='Aggregate results (mean and std)')
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    log_files = log_dir.glob('*_patch-*.log')
    
    results = []
    
    for log_file in log_files:
        val_accs, test_accs = parse_log(log_file)
        if args.agg: 
            results.append({
                'file': str(log_file),
                "dataset": log_file.stem.split('_patch')[0],
                "config": "patch-" + log_file.stem.split('_patch-')[1],
                "val_acc": float(np.mean(val_accs)) * 100 if len(val_accs) > 0 else 0,
                "val_acc_std": float(np.std(val_accs)) * 100 if len(val_accs) > 0 else 0,
                "test_acc": float(np.mean(test_accs)) * 100 if len(test_accs) > 0 else 0,
                "test_acc_std": float(np.std(test_accs)) * 100 if len(test_accs) > 0 else 0,
                "runs": len(test_accs),
                "error": len(test_accs) == 0,
                "has_val": len(val_accs) > 0
            })
        else:
            for i, (val_acc, test_acc) in enumerate(zip(val_accs, test_accs)):
                results.append({
                    'file': str(log_file),
                    "dataset": log_file.stem.split('_patch')[0],
                    "config": "patch-" + log_file.stem.split('_patch-')[1],
                    "val_acc": val_acc * 100,
                    "test_acc": test_acc * 100,
                    "run": i
                })
            

    df = pd.DataFrame(results)
    df = df.sort_values(by=['dataset', 'config'])
    df.to_csv(args.output_file, index=False)
    print(df.to_markdown(index=False))
    print(f"Results saved to {args.output_file}")

    
if __name__ == '__main__':
    main()