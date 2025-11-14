import os
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import average_precision_score

from Detectors import CospyCalibrateDetector
from Datasets import TestDataset, EVAL_DATASET_LIST, EVAL_MODEL_LIST
from utils import seed_torch
from sklearn.metrics import (
    accuracy_score, log_loss, average_precision_score, f1_score,
    roc_auc_score, balanced_accuracy_score, confusion_matrix
)
import numpy as np
import warnings
warnings.filterwarnings("ignore")


class Detector():
    def __init__(self, args):
        super(Detector, self).__init__()

        # Device
        self.device = args.device

        # Initialize the detector
        self.model = CospyCalibrateDetector(
            semantic_weights_path=args.semantic_weights_path,
            artifact_weights_path=args.artifact_weights_path)

        # Load the pre-trained weights
        self.model.load_weights(args.classifier_weights_path)
        self.model.eval()

        # Put the model on the device
        self.model.to(self.device)

    # Prediction function
    def predict(self, inputs):
        inputs = inputs.to(self.device)
        outputs = self.model(inputs)
        prediction = outputs.sigmoid().flatten().tolist()
        return prediction


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Tính ECE (Expected Calibration Error)"""
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob > bins[i]) & (y_prob <= bins[i+1])
        if np.sum(mask) > 0:
            prob_mean = y_prob[mask].mean()
            acc = y_true[mask].mean()
            ece += np.sum(mask) / len(y_true) * abs(acc - prob_mean)
    return ece

def evaluate(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # Metrics
    acc = accuracy_score(y_true, y_pred > 0.5)
    nll = log_loss(y_true, y_pred, eps=1e-7)
    ap = average_precision_score(y_true, y_pred)
    ece = expected_calibration_error(y_true, y_pred)
    f1 = f1_score(y_true, y_pred > 0.5)
    try:
        auc = roc_auc_score(y_true, y_pred)
    except:
        auc = float('nan')
    bacc = balanced_accuracy_score(y_true, y_pred > 0.5)
    # False Negative Rate
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred > 0.5).ravel()
    fnr = fn / (fn + tp) if (fn + tp) > 0 else float('nan')

    return {
        "ACC": acc,
        "NLL": nll,
        "AP": ap,
        "ECE": ece,
        "F1": f1,
        "AUC": auc,
        "bAcc": bacc,
        "FNR": fnr
    }


def test(args):
    # Initialize the detector
    detector = Detector(args)

    # Set the saving directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    save_result_path = os.path.join(args.save_dir, "result.json")
    save_output_path = os.path.join(args.save_dir, "output.json")

    # Begin the evaluation
    result_all = {}
    output_all = {}
    for dataset_name in EVAL_DATASET_LIST:
        result_all[dataset_name] = {}
        output_all[dataset_name] = {}
        for model_name in EVAL_MODEL_LIST:
            test_dataset = TestDataset(dataset=dataset_name, model=model_name, root_path=args.testset_dirpath, transform=detector.model.test_transform)
            test_loader = torch.utils.data.DataLoader(test_dataset,
                                                      batch_size=args.batch_size,
                                                      shuffle=False,
                                                      num_workers=4,
                                                      pin_memory=True)

            # Evaluate the model
            y_pred, y_true = [], []
            for images, labels, _ in tqdm(test_loader, desc=f"Evaluating {dataset_name} {model_name}"):
                y_pred.extend(detector.predict(images))
                y_true.extend(labels.tolist())


            metrics = evaluate(y_pred, y_true)
            print(f"Evaluate on {dataset_name} {model_name} | Size {len(y_true)} | "
                  f"ACC {metrics['ACC']*100:.2f}% | NLL {metrics['NLL']:.4f} | "
                  f"AP {metrics['AP']*100:.2f}% | ECE {metrics['ECE']:.4f} | "
                  f"F1 {metrics['F1']*100:.2f}% | AUC {metrics['AUC']*100:.2f}% | "
                  f"bAcc {metrics['bAcc']*100:.2f}% | FNR {metrics['FNR']*100:.2f}%")
            result_all[dataset_name][model_name] = {"size": len(y_true), **metrics}
            csv_dir = os.path.join(args.save_dir, "csv_outputs")
            os.makedirs(csv_dir, exist_ok=True)
            
            csv_path = os.path.join(csv_dir, f"{dataset_name}_{model_name}.csv")
            
            with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["path_to_image", "true_label", "pred_percentage", "pred_label"])
            
                idx = 0
                for img_path in test_dataset.image_paths:
                    pred_score = float(y_pred[idx])
                    pred_label = 1 if pred_score > 0.5 else 0
                    true_label = int(y_true[idx])
            
                    writer.writerow([
                        img_path,
                        true_label,
                        pred_score,
                        pred_label
                    ])
                    idx += 1
            
            print(f"[CSV SAVED] {csv_path}")
            output_all[dataset_name][model_name] = {"y_pred": y_pred, "y_true": y_true}

    # Save the results
    with open(save_result_path, "w") as f:
        json.dump(result_all, f, indent=4)

    with open(save_output_path, "w") as f:
        json.dump(output_all, f, indent=4)


def scan(args):
    # Initialize the detector
    detector = Detector(args)

    # Define the pre-processing function
    test_transform = detector.model.test_transform

    # Load the image
    image_filepath = input("Please enter the image filepath for scanning: ")
    if not os.path.exists(image_filepath):
        print(f"Image file not found: {image_filepath}")
        image_filepath = input("Please enter the image filepath for scanning: ")

    image = Image.open(image_filepath).convert("RGB")
    image = test_transform(image)
    image = image.unsqueeze(0)
    image = image.to(args.device)
    
    # Make the prediction
    prediction = detector.predict(image)[0]

    if prediction > 0.5:
        print(f"CO-SPY Prediction: {prediction:.3f} - AI-Generated")
    else:
        print(f"CO-SPY Prediction: {prediction:.3f} - Real")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Deep Fake Detection")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--phase", type=str, default="scan", choices=["scan", "test"], help="Phase of the experiment")
    parser.add_argument("--semantic_weights_path", type=str, default="pretrained/semantic_weights.pth", help="Semantic weights path")
    parser.add_argument("--artifact_weights_path", type=str, default="pretrained/artifact_weights.pth", help="Artifact weights path")
    parser.add_argument("--classifier_weights_path", type=str, default="pretrained/classifier_weights.pth", help="Classifier weights path")
    parser.add_argument("--testset_dirpath", type=str, default="data/test", help="Testset directory")
    parser.add_argument("--save_dir", type=str, default="test_results", help="Save directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seed", type=int, default=1024, help="Random seed")

    args = parser.parse_args()

    # Set the random seed
    seed_torch(args.seed)

    # Set the GPU ID
    args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    # Begin the experiment
    if args.phase == "scan":
        scan(args)
    elif args.phase == "test":
        test(args)
    else:
        raise ValueError("Unknown phase")
