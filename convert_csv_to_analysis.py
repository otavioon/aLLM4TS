import pandas as pd
import argparse


def convert_csv_to_analysis(df):
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
        ft_stategy = row["config"].split("_freeze-")[1].split("_")[0].strip()
        if ft_stategy == "2":
            ft_stategy = "Freeze"
        elif ft_stategy == "1":
            ft_stategy = "Partial Freeze"
        else:
            ft_stategy = "Full Finetune"
        
        d = {
            "backbone": "aLLM4TS",
            "tsk_pretext": "aLLM4TS",
            "d_pretext": row["config"].split("_aLLM4TS-")[1].split("_")[0].strip(),
            "head_pred": "MLP",
            "ft_stategy": ft_stategy,
            "tsk_target": "HAR",
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
    args = parser.parse_args()


    df = pd.read_csv(args.input_csv)
    df = convert_csv_to_analysis(df)
    
    if args.output_csv is not None:
        df.to_csv(args.output_csv, index=False)
        print(f"Dataframe saved to {args.output_csv}")
    else:
        print(df.to_markdown(index=False))
        

if __name__ == '__main__':
    main()