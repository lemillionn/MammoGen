
import csv
import matplotlib.pyplot as plt

def read_csv_metrics(csv_path):
    results = {
        'PSNR': [],
        'SSIM': [],
        'Dice': [],
        'SemanticLoss': [],
        'Filenames': []
    }
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            results['Filenames'].append(row['Filename'])
            results['PSNR'].append(float(row['PSNR']) if row['PSNR'] else None)
            results['SSIM'].append(float(row['SSIM']) if row['SSIM'] else None)
            results['Dice'].append(float(row['Dice']) if row['Dice'] else None)
            results['SemanticLoss'].append(float(row['SemanticLoss']) if row['SemanticLoss'] else None)
    return results

def plot_metrics(csv_path):
    results = read_csv_metrics(csv_path)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Evaluation Metrics Over Samples')

    if any(results['PSNR']):
        axs[0,0].plot(results['PSNR'], marker='o')
        axs[0,0].set_title('PSNR')
        axs[0,0].set_ylabel('dB')

    if any(results['SSIM']):
        axs[0,1].plot(results['SSIM'], marker='o', color='orange')
        axs[0,1].set_title('SSIM')

    if any(results['Dice']):
        axs[1,0].plot(results['Dice'], marker='o', color='green')
        axs[1,0].set_title('Dice Coefficient')

    if any(results['SemanticLoss']):
        axs[1,1].plot(results['SemanticLoss'], marker='o', color='red')
        axs[1,1].set_title('Semantic Loss')

    for ax in axs.flat:
        ax.set_xlabel('Sample Index')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    csv_path = "data/outputs/eval_log.csv"
    plot_metrics(csv_path)
