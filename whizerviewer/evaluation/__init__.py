import json
import numpy as np

class WhizReviewerEvaluator:
    def __init__(self, test_data_path):
        self.test_data_path = test_data_path

    def get_score(self, score, pred_score_origin):
        pred_score = [s for s in pred_score_origin if s != 0]
        if len(score) != len(pred_score):
            if len(pred_score) > len(score):
                pred_score = pred_score[-len(score):]
            else:
                pred_score = [3] * (len(score) - len(pred_score)) + pred_score

        min_score = min(score)
        min_score_pred = min(pred_score)
        max_score = max(score)
        max_score_pred = max(pred_score)
        avg_score = sum(score) / len(score)
        avg_score_pred = sum(pred_score) / len(pred_score)

        return min_score, min_score_pred, max_score, max_score_pred, avg_score, avg_score_pred, pred_score

    def get_num_from_context(self, review, name):
        context = review.split(name)[1].split('\n')[0]
        return float(context[0])

    def evaluate(self):
        with open(self.test_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        results = {
            "score": {"min": [], "max": [], "avg": [], "acc": []},
            "soundness": {"min": [], "max": [], "avg": [], "acc": []},
            "presentation": {"min": [], "max": [], "avg": [], "acc": []},
            "contribution": {"min": [], "max": [], "avg": [], "acc": []},
            "decisions": {"true": [], "pred": []}
        }

        error_count = 0

        for item in data.values():
            if item['pred']['paper_desion'] != '':
                # Overall score
                ms1, msp1, maxs1, maxsp1, avgs1, avgsp1, rating_pred = self.get_score(item['rates'], item['pred']['rating'])
                results["score"]["min"].extend([ms1, msp1])
                results["score"]["max"].extend([maxs1, maxsp1])
                results["score"]["avg"].extend([avgs1, avgsp1])
                results["score"]["acc"].append((np.array(item['rates']) == np.array(rating_pred)).all())

                # Soundness
                soundness = [self.get_num_from_context(context['content'], '### Soundness\n\n') for context in item['review_contexts']]
                soundness_pred = [float(i[0]) if i else 0 for i in item['pred']['soundness']]
                ms2, msp2, maxs2, maxsp2, avgs2, avgsp2, soundness_pred = self.get_score(soundness, soundness_pred)
                results["soundness"]["min"].extend([ms2, msp2])
                results["soundness"]["max"].extend([maxs2, maxsp2])
                results["soundness"]["avg"].extend([avgs2, avgsp2])
                results["soundness"]["acc"].append((np.array(soundness) == np.array(soundness_pred)).all())

                # Presentation
                presentation = [self.get_num_from_context(context['content'], '### Presentation\n\n') for context in item['review_contexts']]
                presentation_pred = [float(i[0]) if i else 0 for i in item['pred']['presentation']]
                ms4, msp4, maxs4, maxsp4, avgs4, avgsp4, presentation_pred = self.get_score(presentation, presentation_pred)
                results["presentation"]["min"].extend([ms4, msp4])
                results["presentation"]["max"].extend([maxs4, maxsp4])
                results["presentation"]["avg"].extend([avgs4, avgsp4])
                results["presentation"]["acc"].append((np.array(presentation) == np.array(presentation_pred)).all())

                # Contribution
                contribution = [self.get_num_from_context(context['content'], '### Contribution\n\n') for context in item['review_contexts']]
                contribution_pred = [float(i[0]) if i else 0 for i in item['pred']['contribution']]
                ms, msp, maxs, maxsp, avgs, avgsp, contribution_pred = self.get_score(contribution, contribution_pred)
                results["contribution"]["min"].extend([ms, msp])
                results["contribution"]["max"].extend([maxs, maxsp])
                results["contribution"]["avg"].extend([avgs, avgsp])
                results["contribution"]["acc"].append((np.array(contribution) == np.array(contribution_pred)).all())

                # Decisions
                results["decisions"]["true"].append(1 if 'Accept' in item['decision'].split('\n\n')[1] else 0)
                results["decisions"]["pred"].append(1 if 'accept' == item['pred']['paper_desion'][:6].lower() else 0)
            else:
                error_count += 1

        # Calculate final metrics
        final_results = {
            "total_samples": len(results["decisions"]["true"]),
            "error_count": error_count,
            "accept_rate": {
                "true": np.mean(results["decisions"]["true"]) * 100,
                "pred": np.mean(results["decisions"]["pred"]) * 100
            },
            "decision_accuracy": np.mean(np.array(results["decisions"]["true"]) == np.array(results["decisions"]["pred"])) * 100,
            "score": self.calculate_metrics(results["score"]),
            "soundness": self.calculate_metrics(results["soundness"]),
            "presentation": self.calculate_metrics(results["presentation"]),
            "contribution": self.calculate_metrics(results["contribution"])
        }

        return final_results

    def calculate_metrics(self, category_results):
        true_values = category_results["min"][::2]  # Every other element is a true value
        pred_values = category_results["min"][1::2]  # Every other element is a predicted value
        return {
            "min": {
                "accuracy": np.mean(np.array(true_values) == np.array(pred_values)) * 100,
                "abs_difference": np.mean(np.abs(np.array(true_values) - np.array(pred_values)))
            },
            "max": {
                "accuracy": np.mean(np.array(category_results["max"][::2]) == np.array(category_results["max"][1::2])) * 100,
                "abs_difference": np.mean(np.abs(np.array(category_results["max"][::2]) - np.array(category_results["max"][1::2])))
            },
            "avg": {
                "accuracy": np.mean(np.array(category_results["avg"][::2]) == np.array(category_results["avg"][1::2])) * 100,
                "abs_difference": np.mean(np.abs(np.array(category_results["avg"][::2]) - np.array(category_results["avg"][1::2])))
            },
            "perfect_match": np.mean(category_results["acc"]) * 100
        }

# Example usage
if __name__ == "__main__":
    evaluator = WhizReviewerEvaluator(test_data_path="path/to/your/test/data.json")
    results = evaluator.evaluate()
    print(json.dumps(results, indent=2))