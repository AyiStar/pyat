import typing
import collections

import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict



"""
log = {
    user_no: {
        'labels': Dict[int, int],
        'predictions': Dict[int, float],
        'train_set': List[int],
        'test_set': List[int],
        'model': state_dict
    }
    for user_no in all_users
}
"""


class Analyzer(object):

    def __init__(self, cfg: DictConfig, log: typing.Dict):
        self.cfg = cfg
        self.log = log

    def get_calibrations(self, num_bins) -> typing.Dict:
        preds = []
        labels = []
        confidence = []
        for user_no, user_log in self.log.items():
            for item_no in user_log['test_set']:
                output = user_log['predictions'][item_no]
                confidence.append(output)
                labels.append(user_log['labels'][item_no])
                preds.append(1)
        return self.compute_calibration(np.array(labels), np.array(preds), np.array(confidence), num_bins)

    def get_entropies(self):
        # model: BaseModel = pyat.model.build_base_model(self.cfg.base_model.class_name, self.cfg.base_model)
        # if 'meta_model' in self.cfg:
        #     model: MetaModel = pyat.model.build_meta_model(self.cfg.meta_model.class_name, model, self.cfg.meta_model)
        # model.to(self.cfg.device)
        assert self.cfg.meta_model.name == 'abml'

        entropies = []
        for user_no, user_log in self.log.items():
            log_std = user_log['model']['log_std.user_params'].cpu().numpy()
            entropy = np.sum(log_std + 1.5 + np.log2(np.pi), axis=None, keepdims=False)
            entropies.append(entropy)
        return entropies

    @staticmethod
    def compute_calibration(true_labels, pred_labels, confidences, num_bins=10):
        """Collects predictions into bins used to draw a reliability diagram.
        Arguments:
            true_labels: the true labels for the test examples
            pred_labels: the predicted labels for the test examples
            confidences: the predicted confidences for the test examples
            num_bins: number of bins
        The true_labels, pred_labels, confidences arguments must be NumPy arrays;
        pred_labels and true_labels may contain numeric or string labels.
        For a multi-class model, the predicted label and confidence should be those
        of the highest scoring class.
        Returns a dictionary containing the following NumPy arrays:
            accuracies: the average accuracy for each bin
            confidences: the average confidence for each bin
            counts: the number of examples in each bin
            bins: the confidence thresholds for each bin
            avg_accuracy: the accuracy over the entire test set
            avg_confidence: the average confidence over the entire test set
            expected_calibration_error: a weighted average of all calibration gaps
            max_calibration_error: the largest calibration gap across all bins
        """
        assert (len(confidences) == len(pred_labels))
        assert (len(confidences) == len(true_labels))
        assert (num_bins > 0)

        bin_size = 1.0 / num_bins
        bins = np.linspace(0.0, 1.0, num_bins + 1)
        indices = np.digitize(confidences, bins, right=True)

        bin_accuracies = np.zeros(num_bins, dtype=np.float)
        bin_confidences = np.zeros(num_bins, dtype=np.float)
        bin_counts = np.zeros(num_bins, dtype=np.int)

        for b in range(num_bins):
            selected = np.where(indices == b + 1)[0]
            if len(selected) > 0:
                bin_accuracies[b] = np.mean(true_labels[selected] == pred_labels[selected])
                bin_confidences[b] = np.mean(confidences[selected])
                bin_counts[b] = len(selected)

        avg_acc = np.sum(bin_accuracies * bin_counts) / np.sum(bin_counts)
        avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)

        gaps = np.abs(bin_accuracies - bin_confidences)
        ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
        mce = np.max(gaps)

        return {"accuracies": bin_accuracies,
                "confidences": bin_confidences,
                "counts": bin_counts,
                "bins": bins,
                "avg_accuracy": avg_acc,
                "avg_confidence": avg_conf,
                "expected_calibration_error": ece,
                "max_calibration_error": mce}

    @staticmethod
    def reliability_diagram_subplot(ax, bin_data,
                                    draw_ece=True,
                                    draw_bin_importance=False,
                                    title="Reliability Diagram",
                                    xlabel="Confidence",
                                    ylabel="Expected Accuracy",
                                    bwidth=2,
                                    font_sizes={},
                                    ):
        """Draws a reliability diagram into a subplot."""
        accuracies = bin_data["accuracies"]
        confidences = bin_data["confidences"]
        counts = bin_data["counts"]
        bins = bin_data["bins"]

        bin_size = 1.0 / len(counts)
        positions = bins[:-1] + bin_size / 2.0

        widths = bin_size
        alphas = 0.3
        min_count = np.min(counts)
        max_count = np.max(counts)
        normalized_counts = (counts - min_count) / (max_count - min_count)

        if draw_bin_importance == "alpha":
            alphas = 0.2 + 0.8 * normalized_counts
        elif draw_bin_importance == "width":
            widths = 0.1 * bin_size + 0.9 * bin_size * normalized_counts

        colors = np.zeros((len(counts), 4))
        colors[:, 0] = 240 / 255.
        colors[:, 1] = 60 / 255.
        colors[:, 2] = 60 / 255.
        colors[:, 3] = alphas

        gap_plt = ax.bar(positions, np.abs(accuracies - confidences),
                         bottom=np.minimum(accuracies, confidences), width=widths,
                         edgecolor=colors, color=colors, linewidth=1, label="Gap")

        acc_plt = ax.bar(positions, 0, bottom=accuracies, width=widths,
                         edgecolor="black", color="black", alpha=1.0, linewidth=3,
                         label="Accuracy")

        ax.set_aspect("equal")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")

        if draw_ece:
            ece = (bin_data["expected_calibration_error"] * 100)
            ax.text(0.98, 0.02, "ECE=%.2f" % ece, color="black",
                    ha="right", va="bottom", transform=ax.transAxes, fontsize=font_sizes.get('ece', 24))

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', labelsize=font_sizes.get('ticks', 20))
        ax.tick_params(axis='y', labelsize=font_sizes.get('ticks', 20))
        for k in ['top', 'bottom', 'left', 'right']:
            ax.spines[k].set_linewidth(bwidth)
        # ax.set_xticks(bins)

        ax.set_title(title, fontsize=font_sizes.get('title', 24))
        ax.set_xlabel(xlabel, fontsize=font_sizes.get('xlabel', 24))
        ax.set_ylabel(ylabel, fontsize=font_sizes.get('ylabel', 24))

        ax.legend(handles=[gap_plt, acc_plt], fontsize=font_sizes.get('legend', 24))

    @staticmethod
    def confidence_histogram_subplot(ax, bin_data,
                                     draw_averages=True,
                                     title="Examples per bin",
                                     xlabel="Confidence",
                                     ylabel="Count",
                                     bwidth=2,
                                     font_sizes={}):
        """Draws a confidence histogram into a subplot."""
        counts = bin_data["counts"]
        bins = bin_data["bins"]

        bin_size = 1.0 / len(counts)
        positions = bins[:-1] + bin_size / 2.0

        ax.bar(positions, counts, width=bin_size * 0.9)

        ax.set_xlim(0, 1)
        ax.set_title(title, fontsize=font_sizes.get('title', 24))
        ax.set_xlabel(xlabel, fontsize=font_sizes.get('xlabel', 24))
        ax.set_ylabel(ylabel, fontsize=font_sizes.get('ylabel', 24))
        for k in ['top', 'bottom', 'left', 'right']:
            ax.spines[k].set_linewidth(bwidth)

        if draw_averages:
            acc_plt = ax.axvline(x=bin_data["avg_accuracy"], ls="solid", lw=3,
                                 c="black", label="Accuracy")
            conf_plt = ax.axvline(x=bin_data["avg_confidence"], ls="dotted", lw=3,
                                  c="#444", label="Avg. confidence")
            ax.legend(handles=[acc_plt, conf_plt])

    @staticmethod
    def ecdf_entropy_subplot(ax, entropy_datas,
                             labels,
                             bwidth=2,
                             linewidth=3,
                             title="Empirical CDF of Posterior Entropy",
                             xlabel="Entropy of Estimate Posterior",
                             ylabel="Probability",
                             font_sizes={}):
        """Draws an entropy eCDF into a subplot."""

        assert len(entropy_datas) == len(labels)

        for data, label in zip(entropy_datas, labels):
            ax.plot(np.sort(data), np.linspace(0, 1, len(data), endpoint=False), label=label, linewidth=linewidth)

        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', labelsize=font_sizes.get('ticks', 20))
        ax.tick_params(axis='y', labelsize=font_sizes.get('ticks', 20))
        ax.legend(fontsize=font_sizes.get('legend', 24))
        ax.set_title(title, fontsize=font_sizes.get('title', 24))
        ax.set_xlabel(xlabel, fontsize=font_sizes.get('xlabel', 24))
        ax.set_ylabel(ylabel, fontsize=font_sizes.get('ylabel', 24))
        for k in ['top', 'bottom', 'left', 'right']:
            ax.spines[k].set_linewidth(bwidth)
