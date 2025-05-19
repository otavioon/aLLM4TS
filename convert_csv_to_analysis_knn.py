import pandas as pd
import argparse

training_point = ""


def parse_logs(df):
    lines = []
    df["accuracy"] = df["accuracy"].astype(float)
    for (name, n_neighborns), grouped_df in df.groupby(["name", "n_neighbors"]):
        metric_mean = grouped_df["accuracy"].mean()
        
        line = grouped_df.iloc[0].to_dict()
        line["runs"] = len(grouped_df)
        line["accuracy"] = metric_mean
        
        if len(grouped_df) < 3:
            print(f"Warning (SKIPING): Not enough runs for {name} with {n_neighborns} neighbors. Only {len(grouped_df)} runs found.")
        else:
            lines.append(line) 
        
    df = pd.DataFrame(lines) 
    return df


def convert_csv_to_analysis(df):
    global training_point
    lines = []
    dataset_map = {
        "KuHar": "KH",
        "MotionSense": "MS",
        "RealWorld_thigh": "RW-Thigh",
        "RealWorld_waist": "RW-Waist",
        "UCI": "UCI",
        "WISDM": "WISDM"
    }
    
    print(f"Training point: {training_point}")
    
    
    for (row_idx, row) in df.iterrows():
        # Skip non-normalized rows
        if row["normalization"] == "no":
            continue

        if training_point == "after":
            ft_strategy = row["ft_strategy"]
            if ft_strategy == 2:
                ft_strategy = "Freeze"
            elif ft_strategy == 1:
                ft_strategy = "Partial Freeze"
            else:
                ft_strategy = "Full Finetune"
        else:
            ft_strategy = "Freeze"
            
        n_neighbors = row["n_neighbors"]
        head = f"kNN-{n_neighbors}"
        
        d = {
            "backbone": "aLLM4TS",
            "tsk_pretext": "aLLM4TS",
            "d_pretext": row["pretrain_dataset"],
            "head_pred": head,
            "ft_strategy": ft_strategy,
            "tsk_target": "HAR",
            "d_target": dataset_map[row["dataset_name"]],
            "frac_dtarget": float(row["percent"]) / 100,
            "metric_target": "accuracy",
            "metric": float(row["accuracy"]),
        }
        
        lines.append(d)
    
    df = pd.DataFrame(lines)
    # df = df[df["ft_stategy"] == "Freeze"]
    print(f"Dataframe generated! Shape: {df.shape}")
    
    return df


def main():
    global training_point
    parser = argparse.ArgumentParser(description='Convert CSV to analysis format')
    parser.add_argument('--input_csv', type=str, help='Input CSV file', required=True)
    parser.add_argument('--output_csv', type=str, help='Output CSV file', default=None, required=False)
    args = parser.parse_args()

    if "after" in args.input_csv:
        training_point = "after"
    elif "before" in args.input_csv:
        training_point = "before"
    else:
        raise ValueError("Input CSV file name must contain 'after' or 'before' to indicate the training point.")


    df = pd.read_csv(args.input_csv)
    df = parse_logs(df)
    df = convert_csv_to_analysis(df)
    
    if args.output_csv is not None:
        df.to_csv(args.output_csv, index=False)
        print(f"Dataframe saved to {args.output_csv}")
    else:
        print(df.to_markdown(index=False))
        

if __name__ == '__main__':
    main()