from collections import Counter
from typing import Any

import numpy as np
import sacrebleu
from bert_score import score
from litellm import embedding as litellm_embedding
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from sentence_transformers import SentenceTransformer, util

from explain.eval.score.score_util import get_primitives_from_structure_hypothesis
from explain.llm import create_client


class AccuracyEvaluator:
    """
    Checks the syntax and schema gates of the generated explanations.
    """

    def __init__(self,  **kwargs):
        self.embedding_client = create_client(provider='openai')
        self.embedding = self.embedding_client.client.sync_client.embeddings
        pass


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
            gt_text = gt_text[:-1] if gt_text[-1] == '' else gt_text
            gen_text = gen_text[:-1] if gen_text[-1] == '' else gen_text
        else:
            raise ValueError(f"Invalid type: {type}")
        ref_para = " ".join(gt_text)
        hyp_para = " ".join(gen_text)
        # BLEU
        bleu_metric = BLEU(effective_order=True)
        if len(gen_text) == 0:
            bleu_score = 0.0
        else:
            try:
                bleu_score = bleu_metric.corpus_score(gen_text, [gt_text]).score/100
            except Exception:
                bleu_score = 0.0
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

        for r, h in zip(gt_text, gen_text, strict=False):
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

        for key in token_score.keys():
            if np.isnan(token_score[key]):
                token_score[key] = 0
            else:
                token_score[key] = round(token_score[key], 4)

        return token_score


    def primitive_level_f1(self, gt_structure_hypothesis: str, gen_structure_hypothesis: str) -> float:
        """
        Check whether the generated primitives are in the allowed set.
        """
        gt_primitives = get_primitives_from_structure_hypothesis(gt_structure_hypothesis)
        gen_primitives = get_primitives_from_structure_hypothesis(gen_structure_hypothesis)

        # Compute F1 score for multisets
        gt_counter = Counter(gt_primitives)
        gen_counter = Counter(gen_primitives)
        # True positives: sum of min count for each primitive in both
        true_positives = sum(min(gt_counter[p], gen_counter[p]) for p in gt_counter)
        false_positives = sum(max(0, gen_counter[p] - gt_counter[p]) for p in gen_counter)
        false_negatives = sum(max(0, gt_counter[p] - gen_counter[p]) for p in gt_counter)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return round(f1, 4)

    def get_openai_embedding(self, text, model='text-embedding-3-large'):

        response = litellm_embedding(model=model, input=text)
        return response.data[0].embedding

    def argument_similarity(self, gt_text: str, gen_text: str, mode='BERTScore') -> float:
        """
        Check whether the generated primitives are in the allowed set.
        mode: BERTScore (slow) or SentenceTransformers (fast)
        """
        # Base model: roberta-large
        if mode == 'BERTScore':
            precision, recall, f1 = score([gt_text], [gen_text], lang='en')
            return round(f1.item(), 4)


        elif mode == 'SentenceTransformers':
            model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")  # strong 768-d encoder

            emb1 = model.encode(gt_text, convert_to_tensor=True)
            emb2 = model.encode(gen_text, convert_to_tensor=True)

            cos_sim = util.cos_sim(emb1, emb2).item()
            return round(cos_sim, 4)

        elif mode == 'openai_embeddings':
            emb_ref = self.get_openai_embedding(gt_text)
            emb_gen = self.get_openai_embedding(gen_text)
            cos_sim = util.cos_sim(emb_ref, emb_gen).item()
            return round(cos_sim, 4)
