from typing import Any
from sacrebleu.metrics import BLEU
import sacrebleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import numpy as np


from explain.eval.score._base import Evaluator



class AccuracyEvaluator(Evaluator):
    """
    Checks the syntax and schema gates of the generated explanations.
    """
    
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)


    def token_based_accuracy(self, gt_text: str, gen_text: str, type:str) -> tuple[float, dict[str, Any]]:
        """
        Check whether the generated primitives are in the allowed set.
        """
        token_score = {}

        if type == 'structure_hypothesis':
            gt_text = gt_text.split('\n')
            gen_text = gen_text.split('\n')
        elif type == 'paragraph':
            gt_text = gt_text.split('.')
            gen_text = gen_text.split('.')
        else:
            raise ValueError(f"Invalid type: {type}")
        ref_para = " ".join(gt_text)
        hyp_para = " ".join(gen_text)
        # BLEU
        bleu_metric = BLEU(effective_order=True)
        bleu_score = bleu_metric.corpus_score(gt_text, [gen_text]).score/100
        # ROUGE micro: concatenate sentences into one string each
        rouge_scorer_micro = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        rouge_score = rouge_scorer_micro.score(ref_para, hyp_para)
        # METEOR micro: join into single token lists
        meteor_micro = meteor_score([ref_para.split()], hyp_para.split())

        # Average of sentence-wise scores (macro)
        rouge_scorer_macro = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        meteor_vals, rouge1_vals, rouge2_vals, rougeL_vals = [], [], [], []
        # sentence-level BLEU (sacrebleu.sentence_bleu) then average
        bleu_vals = []

        for r, h in zip(gt_text, gen_text):
            # sentence BLEU
            bleu_vals.append(sacrebleu.sentence_bleu(h, [r]).score)

            # sentence ROUGE
            r_score = rouge_scorer_macro.score(r, h)
            rouge1_vals.append(r_score["rouge1"].fmeasure)
            rouge2_vals.append(r_score["rouge2"].fmeasure)
            rougeL_vals.append(r_score["rougeL"].fmeasure)

            # sentence METEOR
            meteor_vals.append(meteor_score([r.split()], h.split()))

        token_score['bleu_micro'] = bleu_score
        token_score['rouge1_micro'] = rouge_score['rouge1'].fmeasure
        token_score['rouge2_micro'] = rouge_score['rouge2'].fmeasure
        token_score['rougeL_micro'] = rouge_score['rougeL'].fmeasure
        token_score['meteor_micro'] = meteor_micro

        token_score['bleu_macro'] = float(np.mean(bleu_vals))/100
        token_score['rouge1_macro'] = float(np.mean(rouge1_vals))
        token_score['rouge2_macro'] = float(np.mean(rouge2_vals))
        token_score['rougeL_macro'] = float(np.mean(rougeL_vals))
        token_score['meteor_macro'] = float(np.mean(meteor_vals))

        return token_score



