import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def parse_log(log_file):
    accs = []
    with open(log_file, 'r') as f:
        for line in f:
            if "accuracy:" in line:
                acc = float(line.split('accuracy:')[1].strip())
                accs.append(acc)
    return accs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='logs/classification')
    parser.add_argument("--output_file", type=str, default='results.csv')
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    log_files = log_dir.glob('*_patch-*.log')
    
    results = []
    
    for log_file in log_files:
        accs = parse_log(log_file)
        results.append({
            'file': str(log_file),
            "dataset": log_file.stem.split('_patch')[0],
            "config": "patch-" + log_file.stem.split('_patch-')[1],
            "test_acc": float(np.mean(accs)) * 100,
            "test_acc_std": float(np.std(accs)) * 100,
            "runs": len(accs),
            "error": len(accs) == 0
        })
        

    df = pd.DataFrame(results)
    df = df.sort_values(by=['dataset', 'config'])
    df.to_csv(args.output_file, index=False)
    print(df.to_markdown(index=False))
    print(f"Results saved to {args.output_file}")

    
if __name__ == '__main__':
    main()