import pandas as pd
import argparse


def convert_csv_to_analysis(df, backbone="aLLM4TS", tsk_pretext="aLLM4TS", tsk_target="HAR"):
    lines = []
    dataset_map = {
        "KuHar": "KH",
        "MotionSense": "MS",
        "RealWorld_thigh": "RW-Thigh",
        "RealWorld_waist": "RW-Waist",
        "UCI": "UCI",
        "WISDM": "WISDM"
    }
    
    for (row_idx, row) in df.iterrows():
        # Skip non-normalized rows
        norm = row["config"].split("_norm-")[1].split("_")[0].strip()
        if norm == "no":
            continue    
        
        ft_strategy = row["config"].split("_freeze-")[1].split("_")[0].strip()
        if ft_strategy == "2":
            ft_strategy = "Freeze"
        elif ft_strategy == "1":
            ft_strategy = "Partial Freeze"
        else:
            ft_strategy = "Full Finetune"
            
            
        if "_head-" in row["config"]:
            head = row["config"].split("_head-")[1].split("_")[0].strip()
        else:
            head = "aLLM4TS"
        
        d = {
            "backbone": backbone,
            "tsk_pretext": tsk_pretext,
            "d_pretext": row["config"].split("_aLLM4TS-")[1].split("_")[0].strip(),
            "head_pred": head,
            "ft_strategy": ft_strategy,
            "tsk_target": tsk_target,
            "d_target": dataset_map[row["dataset"]],
            "frac_dtarget": float(row["config"].split("_percent-")[1].split("_")[0].strip()) / 100,
            "metric_target": "accuracy",
            "metric": float(row["test_acc"]) / 100,
        }
        
        lines.append(d)
    
    df = pd.DataFrame(lines)
    # df = df[df["ft_stategy"] == "Freeze"]
    print(f"Dataframe generated! Shape: {df.shape}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Convert CSV to analysis format')
    parser.add_argument('--input_csv', type=str, help='Input CSV file', required=True)
    parser.add_argument('--output_csv', type=str, help='Output CSV file', default=None, required=False)
    parser.add_argument("--backbone", type=str, default="aLLM4TS", help='Backbone model name')
    parser.add_argument("--tsk_pretext", type=str, default="aLLM4TS", help='Task pretext model name')
    parser.add_argument("--tsk_target", type=str, default="HAR", help='Task target name')
    args = parser.parse_args()


    df = pd.read_csv(args.input_csv)
    df = convert_csv_to_analysis(df, backbone=args.backbone, tsk_pretext=args.tsk_pretext, tsk_target=args.tsk_target)
    
    if args.output_csv is not None:
        df.to_csv(args.output_csv, index=False)
        print(f"Dataframe saved to {args.output_csv}")
    else:
        print(df.to_markdown(index=False))
        

if __name__ == '__main__':
    main()