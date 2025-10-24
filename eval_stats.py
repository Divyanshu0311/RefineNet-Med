import pandas as pd
import matplotlib.pyplot as plt

def eval_performance(file_path):
    df = pd.read_csv(file_path)
    # print(df.describe())
    # print(df.mean())
    # print("Std deviation is ", df.std())
    return df.mean()

def main():
    mean = []
    ### format is (8, 16, 256) === (sup_epochs, base_filters, img_size)
    mean.append(eval_performance("outputs\eval_results\evaluation_metrics.csv"))
    mean.append(eval_performance("eval_outputs\eval_results\evaluation_metrics.csv"))
    mean.append(eval_performance("test_outputs\evaluation_metrics_32.csv"))
    mean.append(eval_performance("test_outputs\evaluation_metrics_48.csv"))

    model_names = ["Model pretrained (8, 16, 256)", "Model with refiner (8, 16, 256)", 
                    "Model with refiner (20, 32, 256)", "Model with refiner (20, 48, 256)"]

    mean_df = pd.DataFrame(mean, index=model_names)

    print(mean_df)
    metrics = mean_df.columns
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        plt.plot(model_names, mean_df[metric], marker='o', linestyle='-', color='b')
        plt.title(f"{metric} Comparison")
        plt.xlabel("Model")
        plt.ylabel(metric)
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.7)
        for i, val in enumerate(mean_df[metric]):
            plt.text(i, val + 0.01, f"{val:.3f}", ha='center', fontsize=10)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()