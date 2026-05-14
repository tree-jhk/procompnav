"""
Property-based Binary Partitioning (PBP) module for CoIN.
Integrates tree search logic with AIUTA's detection framework.
"""

import cv2
import numpy as np
import os
import re
import time
import requests
from itertools import combinations
from typing import Dict, List, Tuple, Optional, Set
from colorama import Fore, init as init_colorama
from sklearn.cluster import KMeans
import torch

from vlfm.utils.pbp_prompts import (
    build_property_prompt,
    build_common_property_prompt,
    build_text_goal_property_prompt,
    build_text_goal_candidate_yesno_prompt,
    build_mllm_property_prompt,
    build_yesno_property_prompt,
    build_final_verification_prompt,
    build_fallback_selection_prompt
)
from vlfm.vlm.server_wrapper import send_request

init_colorama(autoreset=True)


# ===== Helper functions (from tree_search.py) =====

def decide_from_likelihood(likelihood: list) -> bool:
    """
    Decide Yes/No from likelihood distribution.
    From tree_search.py Line 118-134.
    
    Args:
        likelihood: [['Yes', prob, token_id], ['No', prob, token_id]] or [('Yes', prob), ('No', prob)]
    
    Returns:
        True for Yes, False for No
    """
    yes_p = 0.0
    no_p = 0.0
    has_yes = False
    has_no = False
    for item in likelihood:
        if len(item) == 3:
            label, prob, _ = item
        else:
            label, prob = item
        label = str(label).strip().lower()
        prob = float(prob)
        if label == "yes":
            yes_p += prob
            has_yes = True
        elif label == "no":
            no_p += prob
            has_no = True
    print(f"Likelihoods: {{'yes': {yes_p}, 'no': {no_p}}}")
    if not has_yes or not has_no:
        raise KeyError(f"[ERROR] Missing 'yes' or 'no' key in likelihood: {likelihood}")

    return yes_p >= no_p


def parse_property(output_text: str) -> Optional[str]:
    """
    Parse property from LLM output.
    From tree_search.py Line 345-352 (exact copy).
    """
    prop_match = re.search(r"PROPERTY:\s*(.*)", output_text, re.IGNORECASE)
    property_text = prop_match.group(1).strip() if prop_match else None

    if property_text:
        property_text = re.sub(r"[^a-zA-Z0-9\s]", "", property_text).strip()

    return property_text

def parse_property_list(output_text: str) -> List[str]:
    lines = [line.strip().lower() for line in (output_text or "").splitlines()]
    return [line for line in lines if line]

def l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / (norms + 1e-12)

# ===== Main PBP Class =====

class PropertyBasedBinaryPartitioning:
    """
    Property-based binary partitioning for target object selection.
    """
    
    def __init__(
        self,
        llm_client,
        mllm_client,
        vlm_oracle,
        pbp_config: Dict
    ):
        """
        Initialize PBP module.
        
        Args:
            llm_client: OpenAILLMClient instance
            mllm_client: LLavaNextClient instance
            vlm_oracle: VLMOracle instance
            pbp_config: Dict with trigger_step, max_depth, max_stuck_count, timeout
        """
        self.llm_client = llm_client
        self.mllm_client = mllm_client
        self.vlm_oracle = vlm_oracle
        self.pbp_config = pbp_config
        
        self.max_depth = pbp_config.get('max_depth', 10)
        self.max_stuck_count = pbp_config.get('max_stuck_count', 5)
        self.max_same_property_streak = pbp_config.get("max_same_property_streak", 5)
        self.enable_NLI_based = bool(pbp_config.get("enable_NLI_based", False))
        self.enable_pbp_refinement = bool(pbp_config.get("enable_pbp_refinement", False))
        self.pbp_refinement_thres = float(pbp_config.get("pbp_refinement_thres", 0.5))
        self.instance_grouping_method = str(pbp_config.get("instance_grouping_method", "dense_half_image_text")).lower()
        if self.instance_grouping_method not in [
            "dense_half_image",
            "dense_half_text",
            "dense_half_image_text",
            "dense_average_image",
            "dense_average_text",
            "dense_average_image_text",
            "kmeans",
        ]:
            raise ValueError(
                f"[PBP] Invalid instance_grouping_method: {self.instance_grouping_method}. "
                "Allowed: ['dense_half_image', 'dense_half_text', 'dense_half_image_text', "
                "'dense_average_image', 'dense_average_text', 'dense_average_image_text', 'kmeans']"
            )
        self.nli_kmeans_modal = str(pbp_config.get("NLI_kmeans_modal", "text")).lower()
        if self.nli_kmeans_modal not in ["text", "image"]:
            raise ValueError(f"[PBP] Invalid NLI_kmeans_modal: {self.nli_kmeans_modal}. Allowed: ['text', 'image']")
        self.timeout = pbp_config.get('timeout', 60)
        self.final_verification_mode = pbp_config.get('final_verification_mode', 'candidate_image_mode')

        env_port = os.environ.get("LLava_PORT")
        if env_port is None:
            raise RuntimeError("LLava_PORT environment variable is not set. Please source config/vlm_env.env.")
        try:
            self.mllm_port = int(env_port)
        except ValueError as exc:
            raise RuntimeError(f"Invalid LLava_PORT value '{env_port}': {exc}") from exc

        self.mllm_url = getattr(self.mllm_client, "url", f"http://localhost:{self.mllm_port}/v1")

        user_sim_env_port = os.environ.get("USER_SIMULATOR_PORT")
        if user_sim_env_port is None:
            raise RuntimeError("USER_SIMULATOR_PORT environment variable is not set. Please source config/vlm_env.env.")
        try:
            self.user_simulator_port = int(user_sim_env_port)
        except ValueError as exc:
            raise RuntimeError(f"Invalid USER_SIMULATOR_PORT value '{user_sim_env_port}': {exc}") from exc

        if self.user_simulator_port == self.mllm_port:
            self.user_simulator_mllm_client = self.mllm_client
        else:
            from vlfm.vlm.llava_next import LLavaNextClient
            self.user_simulator_mllm_client = LLavaNextClient(port=self.user_simulator_port)
        self._text_goal_property_candidates: List[str] = []
        self._text_goal_property_candidates_ep_id: Optional[int] = None
        self._text_goal_property_source: str = ""
        self._text_goal_property_category: str = ""
        self._last_mllm_completion_tokens: Optional[int] = None

    def _build_text_kmeans_embeddings(self, descriptions: List[str]) -> np.ndarray:
        text_only_model = getattr(self, "text_only_model", None)
        if text_only_model is None:
            raise ValueError("[PBP] text_only_model is None while NLI_kmeans_modal='text'.")
        kmeans_embeddings = text_only_model.encode(descriptions, batch_size=16, convert_to_numpy=True)
        if not isinstance(kmeans_embeddings, np.ndarray):
            kmeans_embeddings = np.array(kmeans_embeddings)
        return l2_normalize(kmeans_embeddings)

    def _build_image_kmeans_embeddings(self, candidates: Dict[int, Dict], current_object_ids: List[int]) -> np.ndarray:
        image_only_model = getattr(self, "image_only_model", None)
        image_only_processor = getattr(self, "image_only_processor", None)
        if image_only_model is None or image_only_processor is None:
            raise ValueError("[PBP] image_only_model or image_only_processor is None while NLI_kmeans_modal='image'.")
        representative_images = [candidates[obj_id].get("representative_image") for obj_id in current_object_ids]
        if any(img is None for img in representative_images):
            raise ValueError("[PBP] representative_image is missing while NLI_kmeans_modal='image'.")
        representative_images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in representative_images]
        image_inputs = image_only_processor(images=representative_images, return_tensors="pt").to(image_only_model.device)
        with torch.no_grad():
            image_outputs = image_only_model(**image_inputs)
            image_last_hidden_states = image_outputs[0]
        kmeans_embeddings = image_last_hidden_states[:, 1:, :]
        kmeans_embeddings = kmeans_embeddings.reshape(kmeans_embeddings.shape[0], -1)
        kmeans_embeddings = kmeans_embeddings.float().cpu().numpy()
        return l2_normalize(kmeans_embeddings)

    def _build_kmeans_embeddings(
        self,
        descriptions: List[str],
        candidates: Dict[int, Dict],
        current_object_ids: List[int],
    ) -> np.ndarray:
        if self.nli_kmeans_modal == "text":
            print(Fore.YELLOW + "[PBP] Building text embeddings for KMeans...")
            return self._build_text_kmeans_embeddings(descriptions)
        elif self.nli_kmeans_modal == "image":
            print(Fore.YELLOW + "[PBP] Building image embeddings for KMeans...")
            return self._build_image_kmeans_embeddings(candidates, current_object_ids)
        else:
            raise ValueError(f"[PBP] Invalid NLI_kmeans_modal: {self.nli_kmeans_modal}.")

    def _nli_binary_margin_score(self, logits: torch.Tensor) -> torch.Tensor:
        logit_e = logits[:, 0]
        logit_n = logits[:, 1]
        logit_c = logits[:, 2]
        return torch.sigmoid(logit_e - torch.maximum(logit_n, logit_c))

    def _instance_grouping(
        self,
        candidates: Dict[int, Dict],
        current_object_ids: List[int],
        descriptions: List[str],
    ) -> Tuple[List[int], List[int]]:
        method = self.instance_grouping_method
        if method == "kmeans":
            kmeans_embeddings = self._build_kmeans_embeddings(descriptions, candidates, current_object_ids)
            labels = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(kmeans_embeddings)
            g1_object_ids = [obj_id for obj_id, label in zip(current_object_ids, labels) if label == 0]
            g2_object_ids = [obj_id for obj_id, label in zip(current_object_ids, labels) if label == 1]
            return g1_object_ids, g2_object_ids

        if method in ("dense_half_text", "dense_average_text"):
            text_embeddings = self._build_text_kmeans_embeddings(descriptions)
            similarity_matrix = np.matmul(text_embeddings, text_embeddings.T)
        elif method in ("dense_half_image", "dense_average_image"):
            image_embeddings = self._build_image_kmeans_embeddings(candidates, current_object_ids)
            similarity_matrix = np.matmul(image_embeddings, image_embeddings.T)
        elif method in ("dense_half_image_text", "dense_average_image_text"):
            text_embeddings = self._build_text_kmeans_embeddings(descriptions)
            image_embeddings = self._build_image_kmeans_embeddings(candidates, current_object_ids)
            similarity_matrix = 0.5 * (
                np.matmul(text_embeddings, text_embeddings.T) + np.matmul(image_embeddings, image_embeddings.T)
            )
        else:
            raise ValueError(
                f"[PBP] Invalid instance_grouping_method in _instance_grouping: {method}. "
                "Allowed: ['dense_half_image', 'dense_half_text', 'dense_half_image_text', "
                "'dense_average_image', 'dense_average_text', 'dense_average_image_text', 'kmeans']"
            )
        n_objects = len(current_object_ids)

        if method.startswith("dense_half_"):
            k = (n_objects + 1) // 2
            best_subset_indices = None
            best_weight = -np.inf
            for subset in combinations(range(n_objects), k):
                subset_weight = 0.0
                for i in range(len(subset)):
                    for j in range(i + 1, len(subset)):
                        subset_weight += float(similarity_matrix[subset[i], subset[j]])
                if subset_weight > best_weight or (
                    subset_weight == best_weight and (best_subset_indices is None or subset < best_subset_indices)
                ):
                    best_weight = subset_weight
                    best_subset_indices = subset
            best_subset_index_set = set(best_subset_indices)
        elif method.startswith("dense_average_"):
            if n_objects <= 1:
                return list(current_object_ids), []
            if n_objects == 2:
                return [current_object_ids[0]], [current_object_ids[1]]
            alive_indices = list(range(n_objects))
            degrees = similarity_matrix.sum(axis=1) - np.diag(similarity_matrix)
            pair_sum = 0.5 * float(degrees.sum())
            best_subset_indices = None
            best_density = -np.inf
            while len(alive_indices) > 2:
                min_degree_pos = 0
                min_degree_value = float(degrees[alive_indices[0]])
                for pos in range(1, len(alive_indices)):
                    degree_i = float(degrees[alive_indices[pos]])
                    if degree_i < min_degree_value:
                        min_degree_value = degree_i
                        min_degree_pos = pos
                removed_idx = alive_indices.pop(min_degree_pos)
                removed_degree = float(degrees[removed_idx])
                for idx_i in alive_indices:
                    degrees[idx_i] -= float(similarity_matrix[idx_i, removed_idx])
                pair_sum -= removed_degree
                m = len(alive_indices)
                density = (2.0 * pair_sum) / float(m * (m - 1))
                subset_tuple = tuple(alive_indices)
                if density > best_density or (
                    density == best_density and (best_subset_indices is None or subset_tuple < best_subset_indices)
                ):
                    best_density = density
                    best_subset_indices = subset_tuple
            if best_subset_indices is None:
                best_subset_indices = tuple(alive_indices)
            best_subset_index_set = set(best_subset_indices)

        g1_object_ids = [obj_id for idx, obj_id in enumerate(current_object_ids) if idx in best_subset_index_set]
        g2_object_ids = [obj_id for idx, obj_id in enumerate(current_object_ids) if idx not in best_subset_index_set]
        return g1_object_ids, g2_object_ids

    def _get_text_goal_property_candidates(self, ep_id: int, text_goal: str, category: str) -> List[str]:
        if (
            self._text_goal_property_candidates_ep_id == int(ep_id)
            and self._text_goal_property_source == str(text_goal)
            and self._text_goal_property_category == str(category)
            and len(self._text_goal_property_candidates) > 0
        ):
            return list(self._text_goal_property_candidates)
        prompt = build_text_goal_property_prompt(text_goal=text_goal, category=category)
        parsed_props = parse_property_list(self.llm_client.ask(prompt=prompt))
        text_goal_props: List[str] = []
        seen: Set[str] = set()
        for prop in parsed_props:
            prop_norm = str(prop).strip()
            if (not prop_norm) or (prop_norm in seen):
                continue
            seen.add(prop_norm)
            text_goal_props.append(prop_norm)
            if len(text_goal_props) >= 10:
                break
        if len(text_goal_props) == 0:
            fallback_prop = str(text_goal).strip()
            if not fallback_prop:
                fallback_prop = f"{category} mentioned in text goal"
            text_goal_props = [fallback_prop]
            print(Fore.RED + f"[PBP][text_goal] Empty extracted attributes. Fallback to raw text_goal: [{fallback_prop}]")
        self._text_goal_property_candidates = list(text_goal_props)
        self._text_goal_property_candidates_ep_id = int(ep_id)
        self._text_goal_property_source = str(text_goal)
        self._text_goal_property_category = str(category)
        return list(self._text_goal_property_candidates)

    def _propose_property_with_nli(
        self,
        candidates: Dict[int, Dict],
        current_object_ids: List[int],
        descriptions: List[str],
        used_properties: List[str],
        category: str,
        curr_depth: int,
        nli_only_model,
        nli_only_tokenizer,
        task_type: str = "coin",
        text_goal_property_candidates: Optional[List[str]] = None,
    ) -> Optional[Dict[str, object]]:
        print(Fore.YELLOW + f"[PBP] instance_grouping_method: {self.instance_grouping_method}")
        g1_object_ids, g2_object_ids = self._instance_grouping(candidates, current_object_ids, descriptions)
        if len(g1_object_ids) == 0 or len(g2_object_ids) == 0: # This only happens when the instances are [all identical] (KMeans), for (half_dense) this does not happen. 
            return {
                "fallback_final_selection": True,
                "fallback_reason": "identical_embeddings_or_empty_cluster",
                "fallback_selected_object_id": current_object_ids[0],
                "g1_object_ids": g1_object_ids,
                "g2_object_ids": g2_object_ids,
            }

        g1_descriptions = [candidates[obj_id]['description'] for obj_id in g1_object_ids]
        g2_descriptions = [candidates[obj_id]['description'] for obj_id in g2_object_ids]
        if task_type == "text_goal":
            g1_props = set(text_goal_property_candidates or [])
            g2_props = set()
        else:
            g1_prompt = build_common_property_prompt(g1_descriptions, used_properties, category)
            g1_props = set(parse_property_list(self.llm_client.ask(prompt=g1_prompt)))
            if self.instance_grouping_method != "kmeans":
                g2_props = set()
            else:
                g2_prompt = build_common_property_prompt(g2_descriptions, used_properties, category)
                g2_props = set(parse_property_list(self.llm_client.ask(prompt=g2_prompt)))
        g1_props_list = sorted(g1_props)
        g2_props_list = sorted(g2_props)

        print(Fore.BLUE + f"[PBP] Depth: {curr_depth} Group 1 candidate IDs: {g1_object_ids}")
        print(Fore.BLUE + f"[PBP] Depth: {curr_depth} Group 1 properties: {g1_props}")
        print(Fore.BLUE + f"[PBP] Depth: {curr_depth} Group 2 candidate IDs: {g2_object_ids}")
        print(Fore.BLUE + f"[PBP] Depth: {curr_depth} Group 2 properties: {g2_props}")

        if self.instance_grouping_method != "kmeans":
            g1_only = set(g1_props)
            g2_only = set()
            if len(g1_only) == 0:
                return None
        else:
            g1_only = g1_props - g2_props
            g2_only = g2_props - g1_props
            if not g1_only and not g2_only:
                if len(g1_props) == 0 and len(g2_props) == 0:
                    return None
                if len(g1_props) >= len(g2_props):
                    g1_only = g1_props
                else:
                    g2_only = g2_props

        def avg_delta_scores(prop: str, captions: List[str]) -> Tuple[float, float]:
            deltas = []
            sigs = []
            for i in range(0, len(captions), 16):
                batch_caps = captions[i:i + 16]
                inputs = nli_only_tokenizer(
                    batch_caps,
                    [f"the {category} has the attribute {prop}"] * len(batch_caps),
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(nli_only_model.device) for k, v in inputs.items()}
                outputs = nli_only_model(**inputs)
                logits = outputs.logits
                score = self._nli_binary_margin_score(logits)
                deltas.append(score.detach().cpu())
                sigs.append(score.detach().cpu())
            delta_all = torch.cat(deltas)
            sig_all = torch.cat(sigs)
            return float(delta_all.mean().item()), float(sig_all.mean().item())

        g1_best_prop = None
        g1_best_logit_score = -1e9
        g1_best_prob_score = -1e9
        g1_ranked_props: List[Dict[str, object]] = []
        if g1_only:
            g1_candidates = list(g1_only)
            g1_scores = []
            g1_sig_scores = []
            for p in g1_candidates:
                g1_pos, g1_pos_sig = avg_delta_scores(p, g1_descriptions)
                g1_neg, g1_neg_sig = avg_delta_scores(p, g2_descriptions)
                g1_scores.append(g1_pos - g1_neg)
                g1_sig_scores.append(g1_pos_sig - g1_neg_sig)
            g1_ranked_props = sorted(
                [
                    {
                        "prop": p,
                        "prop_group": "g1",
                        "logit_score": float(score),
                        "prob_score": float(sig_score),
                    }
                    for p, score, sig_score in zip(g1_candidates, g1_scores, g1_sig_scores)
                ],
                key=lambda x: x["logit_score"],
                reverse=True,
            )
            g1_idx = int(np.argmax(g1_scores))
            g1_best_prop = g1_candidates[g1_idx]
            g1_best_logit_score = float(g1_scores[g1_idx])
            g1_best_prob_score = float(g1_sig_scores[g1_idx])

        g2_best_prop = None
        g2_best_logit_score = -1e9
        g2_best_prob_score = -1e9
        g2_ranked_props: List[Dict[str, object]] = []
        if g2_only:
            g2_candidates = list(g2_only)
            g2_scores = []
            g2_sig_scores = []
            for p in g2_candidates:
                g2_pos, g2_pos_sig = avg_delta_scores(p, g2_descriptions)
                g2_neg, g2_neg_sig = avg_delta_scores(p, g1_descriptions)
                g2_scores.append(g2_pos - g2_neg)
                g2_sig_scores.append(g2_pos_sig - g2_neg_sig)
            g2_ranked_props = sorted(
                [
                    {
                        "prop": p,
                        "prop_group": "g2",
                        "logit_score": float(score),
                        "prob_score": float(sig_score),
                    }
                    for p, score, sig_score in zip(g2_candidates, g2_scores, g2_sig_scores)
                ],
                key=lambda x: x["logit_score"],
                reverse=True,
            )
            g2_idx = int(np.argmax(g2_scores))
            g2_best_prop = g2_candidates[g2_idx]
            g2_best_logit_score = float(g2_scores[g2_idx])
            g2_best_prob_score = float(g2_sig_scores[g2_idx])

        if g1_best_prop is None and g2_best_prop is None:
            return None
        if g2_best_logit_score > g1_best_logit_score:
            prop = g2_best_prop
            prop_group = "g2"
        else:
            prop = g1_best_prop
            prop_group = "g1"
        print(Fore.MAGENTA + f"[PBP] {prop_group=} {prop=} (scores: g1_best_logit_score={g1_best_logit_score}, g2_best_logit_score={g2_best_logit_score}, g1_best_prob_score={g1_best_prob_score}, g2_best_prob_score={g2_best_prob_score})")

        return {
            "prop": prop,
            "prop_group": prop_group,
            "g1_object_ids": g1_object_ids,
            "g2_object_ids": g2_object_ids,
            "g1_props_list": g1_props_list,
            "g2_props_list": g2_props_list,
            "g1_ranked_props": g1_ranked_props,
            "g2_ranked_props": g2_ranked_props,
        }

    def _pas_scores( # PAS: Property Alignment Score
        self,
        nli_only_model,
        nli_only_tokenizer,
        captions: List[str],
        category: str,
        prop: str,
    ) -> List[float]:
        scores: List[float] = []
        for i in range(0, len(captions), 16):
            batch_caps = captions[i:i + 16]
            inputs = nli_only_tokenizer(
                batch_caps,
                [f"the {category} has the attribute {prop}"] * len(batch_caps),
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(nli_only_model.device) for k, v in inputs.items()}
            outputs = nli_only_model(**inputs)
            logits = outputs.logits
            score = self._nli_binary_margin_score(logits)
            scores.extend(score.detach().cpu().tolist())
        return [float(x) for x in scores]

    def _pas_for_non_refinement_text_goal(
        self,
        nli_only_model,
        nli_only_tokenizer,
        candidates: Dict[int, Dict],
        current_object_ids: List[int],
        category: str,
        prop: str,
    ) -> Dict[int, float]:
        pas_scores = self._pas_scores(
            nli_only_model=nli_only_model,
            nli_only_tokenizer=nli_only_tokenizer,
            captions=[candidates[obj_id]["description"] for obj_id in current_object_ids],
            category=category,
            prop=prop,
        )
        return {int(obj_id): float(score) for obj_id, score in zip(current_object_ids, pas_scores)}

    def _refine_groups_with_pas(
        self,
        candidates: Dict[int, Dict],
        category: str,
        g1_object_ids: List[int],
        g2_object_ids: List[int],
        prop_candidates: List[Dict[str, object]],
        default_prop: str,
        default_prop_group: str,
        nli_only_model,
        nli_only_tokenizer,
        task_type: str = "coin",
    ) -> Dict[str, object]:
        # Refinement direction depends on attribute source reliability:
        #   text_goal: attributes parsed from the target's own description -> trustworthy ->
        #     allow both N->E (pull missed targets in) and E->N (expel false positives from G_c).
        #   coin: attributes hypothesized by LLM from G_c captions -> noisy ->
        #     N->E only, to avoid expelling the true target before user feedback resolves it.
        allow_extracted_to_non_extracted = (str(task_type).strip().lower() == "text_goal")
        yes_object_ids = g1_object_ids if default_prop_group == "g1" else g2_object_ids
        no_object_ids = g2_object_ids if default_prop_group == "g1" else g1_object_ids
        refinement_before_groups = {
            "g1_object_ids": g1_object_ids,
            "g2_object_ids": g2_object_ids,
            "extracted_group": default_prop_group,
        }
        refinement_after_groups = {
            "g1_object_ids": g1_object_ids,
            "g2_object_ids": g2_object_ids,
            "extracted_group": default_prop_group,
        }
        refinement_pas = []
        mllm_refine_decision = []
        refinement_change = False
        prop = default_prop
        prop_group = default_prop_group

        for cand in prop_candidates:
            cand_prop = str(cand["prop"])
            cand_group = str(cand["prop_group"])
            extracted_ids = g1_object_ids if cand_group == "g1" else g2_object_ids
            non_extracted_ids = g2_object_ids if cand_group == "g1" else g1_object_ids
            extracted_descriptions = [candidates[obj_id]['description'] for obj_id in extracted_ids]
            non_extracted_descriptions = [candidates[obj_id]['description'] for obj_id in non_extracted_ids]
            extracted_group_pas = self._pas_scores(
                nli_only_model=nli_only_model,
                nli_only_tokenizer=nli_only_tokenizer,
                captions=extracted_descriptions,
                category=category,
                prop=cand_prop,
            )
            non_extracted_group_pas = self._pas_scores(
                nli_only_model=nli_only_model,
                nli_only_tokenizer=nli_only_tokenizer,
                captions=non_extracted_descriptions,
                category=category,
                prop=cand_prop,
            )
            non_extracted_move = [obj_id for obj_id, pas in zip(non_extracted_ids, non_extracted_group_pas) if pas >= self.pbp_refinement_thres]
            non_extracted_move_set = set(non_extracted_move)
            if allow_extracted_to_non_extracted:
                extracted_move = [obj_id for obj_id, pas in zip(extracted_ids, extracted_group_pas) if pas < self.pbp_refinement_thres]
            else:
                extracted_move = []
            extracted_move_set = set(extracted_move)
            extracted_keep = [obj_id for obj_id in extracted_ids if obj_id not in extracted_move_set]
            refined_extracted = extracted_keep + non_extracted_move
            refined_non_extracted = [obj_id for obj_id in non_extracted_ids if obj_id not in non_extracted_move_set] + extracted_move

            non_extracted_label = "g2" if cand_group == "g1" else "g1"
            cand_refinement_pas = [
                {
                    "object_id": obj_id,
                    "caption": candidates[obj_id]['description'],
                    "group_before": cand_group,
                    "group_after": non_extracted_label if obj_id in extracted_move_set else cand_group,
                    "pas": float(pas),
                }
                for obj_id, pas in zip(extracted_ids, extracted_group_pas)
            ] + [
                {
                    "object_id": obj_id,
                    "caption": candidates[obj_id]['description'],
                    "group_before": ("g2" if cand_group == "g1" else "g1"),
                    "group_after": cand_group if obj_id in non_extracted_move_set else ("g2" if cand_group == "g1" else "g1"),
                    "pas": float(pas),
                }
                for obj_id, pas in zip(non_extracted_ids, non_extracted_group_pas)
            ]
            refinement_before_groups = {
                "g1_object_ids": g1_object_ids,
                "g2_object_ids": g2_object_ids,
                "extracted_group": cand_group,
            }
            refinement_after_groups = {
                "g1_object_ids": refined_extracted if cand_group == "g1" else refined_non_extracted,
                "g2_object_ids": refined_non_extracted if cand_group == "g1" else refined_extracted,
                "extracted_group": cand_group,
            }
            refinement_pas = cand_refinement_pas
            refinement_change = bool(any(item["group_before"] != item["group_after"] for item in cand_refinement_pas))
            prop = cand_prop
            prop_group = cand_group
            yes_object_ids = refined_extracted
            no_object_ids = refined_non_extracted
            if len(yes_object_ids) > 0 and len(no_object_ids) > 0:
                break

        return {
            "prop": prop,
            "prop_group": prop_group,
            "yes_object_ids": yes_object_ids,
            "no_object_ids": no_object_ids,
            "refinement_before_groups": refinement_before_groups,
            "refinement_after_groups": refinement_after_groups,
            "refinement_pas": refinement_pas,
            "mllm_refine_decision": mllm_refine_decision,
            "refinement_change": refinement_change,
        }

    def group_refinement(
        self,
        candidates: Dict[int, Dict],
        category: str,
        prop: str,
        prop_group: str,
        g1_object_ids: List[int],
        g2_object_ids: List[int],
        g1_ranked_props: List[Dict[str, object]],
        g2_ranked_props: List[Dict[str, object]],
        nli_only_model,
        nli_only_tokenizer,
        task_type: str = "coin",
    ) -> Dict[str, object]:
        yes_object_ids = g1_object_ids if prop_group == "g1" else g2_object_ids # Cohesive set containing property
        no_object_ids = g2_object_ids if prop_group == "g1" else g1_object_ids # Remainder set
        refinement_before_groups = {"g1_object_ids": g1_object_ids, "g2_object_ids": g2_object_ids, "extracted_group": prop_group}
        refinement_after_groups = {"g1_object_ids": g1_object_ids, "g2_object_ids": g2_object_ids, "extracted_group": prop_group}
        refinement_pas = []
        mllm_refine_decision = []
        refinement_change = False
        if self.enable_pbp_refinement:
            refinement_result = self._refine_groups_with_pas(
                candidates=candidates,
                category=category,
                g1_object_ids=g1_object_ids,
                g2_object_ids=g2_object_ids,
                prop_candidates=sorted(
                    g1_ranked_props + g2_ranked_props,
                    key=lambda x: (x["logit_score"], 1 if x["prop_group"] == "g1" else 0),
                    reverse=True,
                ),
                default_prop=prop,
                default_prop_group=prop_group,
                nli_only_model=nli_only_model,
                nli_only_tokenizer=nli_only_tokenizer,
                task_type=task_type,
            )
            prop = str(refinement_result["prop"])
            prop_group = str(refinement_result["prop_group"])
            yes_object_ids = list(refinement_result["yes_object_ids"])
            no_object_ids = list(refinement_result["no_object_ids"])
            refinement_before_groups = dict(refinement_result["refinement_before_groups"])
            refinement_after_groups = dict(refinement_result["refinement_after_groups"])
            refinement_pas = list(refinement_result["refinement_pas"])
            mllm_refine_decision = list(refinement_result["mllm_refine_decision"])
            refinement_change = bool(refinement_result["refinement_change"])
            print(
                Fore.MAGENTA
                + f"[PBP] refinement {prop_group=} {prop=} (thres={self.pbp_refinement_thres}, refinement_change={refinement_change}, yes={len(yes_object_ids)}, no={len(no_object_ids)})"
            )
            print(Fore.BLUE + f"[PBP] refinement before: g1={refinement_before_groups['g1_object_ids']}, g2={refinement_before_groups['g2_object_ids']}")
            print(Fore.BLUE + f"[PBP] refinement after: g1={refinement_after_groups['g1_object_ids']}, g2={refinement_after_groups['g2_object_ids']}")
            moved_object_ids = [item["object_id"] for item in refinement_pas if item["group_before"] != item["group_after"]]
            print(
                (Fore.GREEN if moved_object_ids else Fore.YELLOW)
                + f"[PBP] refinement group change: {'detected' if moved_object_ids else 'none'} (moved_object_ids={moved_object_ids})"
            )
            print(Fore.CYAN + f"[PBP] mllm_refine_decision={mllm_refine_decision}")
            extracted_group = str(refinement_before_groups["extracted_group"])
            non_extracted_group = "g2" if extracted_group == "g1" else "g1"
            should_be_non_extracted = [
                item for item in refinement_pas
                if str(item["group_before"]) == extracted_group and float(item["pas"]) < self.pbp_refinement_thres
            ]
            should_be_extracted = [
                item for item in refinement_pas
                if str(item["group_before"]) == non_extracted_group and float(item["pas"]) >= self.pbp_refinement_thres
            ]
            print(
                Fore.CYAN
                + f"[PBP] refinement mismatch (extracted->{non_extracted_group}): "
                f"{[(item['object_id'], round(float(item['pas']), 4), '<') for item in should_be_non_extracted]} vs thres={self.pbp_refinement_thres}"
            )
            print(
                Fore.CYAN
                + f"[PBP] refinement mismatch ({non_extracted_group}->extracted): "
                f"{[(item['object_id'], round(float(item['pas']), 4), '>=') for item in should_be_extracted]} vs thres={self.pbp_refinement_thres}"
            )
            for item in refinement_pas:
                pas = float(item["pas"])
                item_before = str(item["group_before"])
                thres = self.pbp_refinement_thres
                comp = ">=" if pas >= thres else "<"
                print(
                    Fore.CYAN
                    + f"[PBP] refinement score-check obj_id={item['object_id']} "
                    f"group_before={item['group_before']} group_after={item['group_after']} "
                    f"PAS={pas:.4f} ({comp} {thres})"
                )
        return {
            "prop": prop,
            "prop_group": prop_group,
            "yes_object_ids": yes_object_ids, # Cohesive set containing property
            "no_object_ids": no_object_ids, # Remainder set
            "refinement_before_groups": refinement_before_groups,
            "refinement_after_groups": refinement_after_groups,
            "refinement_pas": refinement_pas,
            "mllm_refine_decision": mllm_refine_decision,
            "refinement_change": refinement_change,
        }

    def _text_goal_candidate_yesno(
        self,
        text_goal: str,
        category: str,
        candidate_caption: str,
        candidate_prop: str,
    ) -> bool:
        prompt = build_text_goal_candidate_yesno_prompt(
            text_goal=text_goal,
            category=category,
            candidate_caption=candidate_caption,
        )
        response_text, likelihood = self.llm_client.ask_with_likelihood(
            prompt=prompt,
            max_tokens=1,
            top_logprobs=20,
        )
        decision = decide_from_likelihood(likelihood)
        print(Fore.CYAN + f"[PBP][text_goal] LLM decision: {'Yes' if decision else 'No'} | response={response_text}")
        return decision

    def _coin_prefilter_with_facts(
        self,
        ep_id: int,
        category: str,
        candidates: Dict[int, Dict],
        current_object_ids: List[int],
        current_step: int,
        trigger_step: int,
        logs: List[Dict],
    ) -> Tuple[List[int], Optional[str]]:
        forced_status: Optional[str] = None
        dialogue_history = self.vlm_oracle.pbp_dialogue_history.get(ep_id, [])
        if len(dialogue_history) == 0:
            return current_object_ids, forced_status

        original_object_ids = current_object_ids.copy()
        filtered_object_ids = current_object_ids.copy()
        nli_only_model = getattr(self, "nli_only_model", None)
        nli_only_tokenizer = getattr(self, "nli_only_tokenizer", None)
        for item in dialogue_history:
            if len(filtered_object_ids) == 0:
                break
            if len(item) < 2:
                continue
            fact_attr = str(item[0]).strip()
            fact_answer = str(item[1]).strip().lower()
            if (not fact_attr) or fact_answer not in ("yes", "no"):
                continue
            fact_captions = [candidates[obj_id]["description"] for obj_id in filtered_object_ids]
            fact_pas_scores = self._pas_scores(
                nli_only_model=nli_only_model,
                nli_only_tokenizer=nli_only_tokenizer,
                captions=fact_captions,
                category=category,
                prop=fact_attr,
            )
            if fact_answer == "yes":
                filtered_object_ids = [obj_id for obj_id, pas in zip(filtered_object_ids, fact_pas_scores) if float(pas) >= float(self.pbp_refinement_thres)]
            else:
                filtered_object_ids = [obj_id for obj_id, pas in zip(filtered_object_ids, fact_pas_scores) if float(pas) < float(self.pbp_refinement_thres)]

        if len(filtered_object_ids) == 0:
            if int(current_step) >= int(trigger_step):
                current_object_ids = original_object_ids
                logs.append({"round": 0, "depth": 0, "next_size": len(current_object_ids), "pbp_round_status": "prefilter_empty_use_original_trigger_step"})
                print(Fore.YELLOW + "[PBP][coin] Fact prefilter became empty, but trigger reached. Reusing original candidates.")
            else:
                forced_status = "PBP_WAIT_FOR_NEW_INSTANCE"
                logs.append({"round": 0, "depth": 0, "next_size": 0, "pbp_round_status": "prefilter_empty_wait_for_new_instance"})
                print(Fore.YELLOW + "[PBP][coin] Fact prefilter became empty. Waiting for new matured instance.")
            return current_object_ids, forced_status

        current_object_ids = filtered_object_ids
        logs.append({"round": 0, "depth": 0, "next_size": len(current_object_ids), "pbp_round_status": "prefilter_applied"})
        print(Fore.CYAN + f"[PBP][coin] Fact prefilter applied: {len(original_object_ids)} -> {len(current_object_ids)}")
        return current_object_ids, forced_status

    def run(
        self,
        ep_id: int,
        category: str,
        target_image: np.ndarray,
        candidates: Dict[int, Dict],
        current_step: int,
        num_frontiers: int,
        task_type: str = "coin",
        text_goal: str = "",
    ) -> Tuple[Optional[int], Dict]:
        """
        Run PBP to select target object from candidates.
        
        Args:
            ep_id: Episode ID
            category: Object category
            target_image: Target object image (for user simulator)
            candidates: {group_unique_id: {diverse_views_image, description, position}}
        
        Returns:
            (selected_object_id, pbp_run_log)
        """
        start_time = time.perf_counter()
        round_times_sec: List[float] = []
        depth_times_sec: List[float] = []
        depth_time_accum_sec = 0.0
        
        # Initialize tracking
        current_object_ids = sorted(candidates.keys()) # [0, 1, 2, ..., N-1] (N candidates)
        used_properties: List[str] = []
        rounds = 0  # attempts
        depth = 0  # tree depth (advances only on successful split)
        max_rounds = int(self.max_depth * max(1, int(self.max_same_property_streak) + 1))
        logs = []
        last_multi_candidate_node = None
        prop_counts: Dict[str, int] = {}
        fallback_reason: Optional[str] = None
        forced_status: Optional[str] = None
        trigger_step = int(self.pbp_config.get("trigger_step", 400))
        coin_question_cap = 4
        task_type = str(task_type).strip().lower()
        if task_type not in ("coin", "text_goal"):
            raise ValueError(f"[PBP] Unsupported task_type='{task_type}'. Expected one of ['coin', 'text_goal'].")
        if task_type == "text_goal" and (not self.enable_NLI_based):
            raise ValueError("[PBP] task_type='text_goal' requires enable_NLI_based=True.")
        text_goal_property_candidates: List[str] = []
        if task_type == "text_goal":
            if (
                self._text_goal_property_candidates_ep_id == int(ep_id)
                and self._text_goal_property_category == str(category)
                and len(self._text_goal_property_candidates) > 0
            ):
                text_goal_property_candidates = list(self._text_goal_property_candidates)
            else:
                fallback_prop = str(text_goal).strip()
                if not fallback_prop:
                    fallback_prop = f"{category} mentioned in text goal"
                text_goal_property_candidates = [fallback_prop]
                print(
                    Fore.RED
                    + f"[PBP][text_goal] Missing precomputed attributes in run(). Fallback to [{fallback_prop}]"
                )
        
        print(Fore.CYAN + f"[PBP] Starting with {len(current_object_ids)} candidates")
        print(Fore.MAGENTA + f"[PBP][TASK] task_type={task_type}")
        if task_type == "coin":
            print(Fore.MAGENTA + "[PBP][TASK] Extraction: shared attributes from cohesive set each round.")
            print(Fore.MAGENTA + "[PBP][TASK] Interaction: ask user simulator yes/no each round.")
        else:
            print(Fore.MAGENTA + f"[PBP][TASK] Extraction: fixed attributes from text_goal (n={len(text_goal_property_candidates)}).")
            print(Fore.MAGENTA + "[PBP][TASK] Interaction: skip user simulator and always choose refined cohesive set.")

        if task_type == "coin" and self.enable_NLI_based:
            current_object_ids, forced_status = self._coin_prefilter_with_facts(
                ep_id=ep_id,
                category=category,
                candidates=candidates,
                current_object_ids=current_object_ids,
                current_step=current_step,
                trigger_step=trigger_step,
                logs=logs,
            )
        
        # Main PBP loop
        while forced_status is None and depth < self.max_depth and rounds < max_rounds:
            rounds += 1
            curr_depth = depth + 1
            
            print(
                Fore.MAGENTA + f"[PBP] Depth: {curr_depth}, Round: {rounds}, Candidates: {current_object_ids}"
            )
            round_start = time.perf_counter()
            
            # Save multi-candidate node
            if len(current_object_ids) >= 2:
                last_multi_candidate_node = current_object_ids.copy()
            
            # Termination check
            if len(current_object_ids) <= 1:
                round_elapsed = float(time.perf_counter() - round_start)
                round_times_sec.append(round_elapsed)
                depth_time_accum_sec += round_elapsed
                print(Fore.GREEN + f"[PBP] Depth: {curr_depth} Termination: single candidate selected.")
                print(Fore.MAGENTA + f"[PBP] Round {rounds} time: {round_elapsed:.6f}s")
                break
            
            # Get descriptions for current candidates
            descriptions = [candidates[obj_id]['description'] for obj_id in current_object_ids]
            refinement_pas = []
            
            # === Property Proposal ===
            used_mllm_fallback = False
            if self.enable_NLI_based:
                if self.instance_grouping_method == "kmeans":
                    print(
                        Fore.YELLOW
                        + f"[PBP] Using NLI-based grouping to propose binary property... "
                        + f"(grouping={self.instance_grouping_method}, modal={self.nli_kmeans_modal})"
                    )
                else:
                    print(
                        Fore.YELLOW
                        + f"[PBP] Using NLI-based grouping to propose binary property... "
                        + f"(grouping={self.instance_grouping_method})"
                    )
                if task_type == "text_goal":
                    print(Fore.MAGENTA + f"[PBP][ROUND {rounds}] Extraction(text_goal): reuse fixed attributes {text_goal_property_candidates}")
                else:
                    print(Fore.MAGENTA + f"[PBP][ROUND {rounds}] Extraction(coin): infer shared attributes from grouped captions.")
                nli_only_model = getattr(self, "nli_only_model", None)
                nli_only_tokenizer = getattr(self, "nli_only_tokenizer", None)
                proposal_result = self._propose_property_with_nli(
                    candidates=candidates,
                    current_object_ids=current_object_ids,
                    descriptions=descriptions,
                    used_properties=used_properties,
                    category=category,
                    curr_depth=curr_depth,
                    nli_only_model=nli_only_model,
                    nli_only_tokenizer=nli_only_tokenizer,
                    task_type=task_type,
                    text_goal_property_candidates=text_goal_property_candidates,
                )
                if proposal_result is None:
                    round_elapsed = float(time.perf_counter() - round_start)
                    round_times_sec.append(round_elapsed)
                    depth_time_accum_sec += round_elapsed
                    print(Fore.MAGENTA + f"[PBP] Round {rounds} time: {round_elapsed:.6f}s")
                    logs.append(
                        {
                            "round": rounds,
                            "depth": curr_depth,
                            "note": "failed_to_find_prop",
                            "round_elapsed_time_sec": round_elapsed,
                            "pbp_round_status": "no_property_found",
                        }
                    )
                    continue
                if bool(proposal_result.get("fallback_final_selection", False)):
                    fallback_selected_object_id = proposal_result["fallback_selected_object_id"]
                    fallback_reason = str(proposal_result.get("fallback_reason", "identical_embeddings_or_empty_cluster"))
                    round_elapsed = float(time.perf_counter() - round_start)
                    round_times_sec.append(round_elapsed)
                    depth_time_accum_sec += round_elapsed
                    print(Fore.YELLOW + f"[PBP] Empty grouping side detected. Force-selecting first active candidate: {fallback_selected_object_id}")
                    print(Fore.MAGENTA + f"[PBP] Round {rounds} time: {round_elapsed:.6f}s")
                    logs.append(
                        {
                            "round": rounds,
                            "depth": curr_depth,
                            "fallback_reason": fallback_reason,
                            "selected_object_id": fallback_selected_object_id,
                            "current_object_ids": list(current_object_ids),
                            "g1_object_ids": list(proposal_result.get("g1_object_ids", [])),
                            "g2_object_ids": list(proposal_result.get("g2_object_ids", [])),
                            "round_elapsed_time_sec": round_elapsed,
                            "pbp_round_status": "fallback_identical_embeddings_or_empty_cluster",
                        }
                    )
                    current_object_ids = [fallback_selected_object_id]
                    break
                prop = str(proposal_result["prop"])
                prop_group = str(proposal_result["prop_group"])
                g1_object_ids = list(proposal_result["g1_object_ids"])
                g2_object_ids = list(proposal_result["g2_object_ids"])
                g1_props_list = list(proposal_result["g1_props_list"])
                g2_props_list = list(proposal_result["g2_props_list"])
                g1_ranked_props = list(proposal_result["g1_ranked_props"])
                g2_ranked_props = list(proposal_result["g2_ranked_props"])

                refinement_result = self.group_refinement(
                    candidates=candidates,
                    category=category,
                    prop=prop,
                    prop_group=prop_group,
                    g1_object_ids=g1_object_ids,
                    g2_object_ids=g2_object_ids,
                    g1_ranked_props=g1_ranked_props,
                    g2_ranked_props=g2_ranked_props,
                    nli_only_model=nli_only_model,
                    nli_only_tokenizer=nli_only_tokenizer,
                    task_type=task_type,
                )
                prop = str(refinement_result["prop"])
                prop_group = str(refinement_result["prop_group"])
                cohesive_object_ids = g1_object_ids if prop_group == "g1" else g2_object_ids
                if task_type == "coin":
                    yes_object_ids = list(refinement_result["yes_object_ids"])
                    no_object_ids = list(refinement_result["no_object_ids"])
                elif task_type == "text_goal":
                    yes_object_ids = list(refinement_result["yes_object_ids"]) # Cohesive set containing property
                    no_object_ids = list(refinement_result["no_object_ids"]) # Remainder set
                refinement_before_groups = dict(refinement_result["refinement_before_groups"])
                refinement_after_groups = dict(refinement_result["refinement_after_groups"])
                if task_type == "coin":
                    if prop_group == "g1":
                        refinement_after_groups = {"g1_object_ids": yes_object_ids, "g2_object_ids": no_object_ids, "extracted_group": "g1"}
                    else:
                        refinement_after_groups = {"g1_object_ids": no_object_ids, "g2_object_ids": yes_object_ids, "extracted_group": "g2"}
                refinement_pas = list(refinement_result["refinement_pas"])
                mllm_refine_decision = list(refinement_result["mllm_refine_decision"])
                refinement_change = bool(refinement_result["refinement_change"])

                log_item = {
                    "round": rounds,
                    "depth": curr_depth,
                    "chosen_property": prop,
                    "prop_group": prop_group,
                    "g1_properties": g1_props_list,
                    "g2_properties": g2_props_list,
                    "g1_property_count": len(g1_props_list),
                    "g2_property_count": len(g2_props_list),
                    "refinement_enabled": bool(self.enable_pbp_refinement),
                    "refinement_threshold": float(self.pbp_refinement_thres),
                    "refinement_change": bool(refinement_change),
                    "refinement_pas": refinement_pas,
                    "mllm_refine_decision": mllm_refine_decision,
                    "refinement_before_groups": refinement_before_groups,
                    "refinement_after_groups": refinement_after_groups,
                }

                if task_type == "text_goal" and (len(yes_object_ids) == 0 or len(no_object_ids) == 0):
                    round_elapsed = float(time.perf_counter() - round_start)
                    round_times_sec.append(round_elapsed)
                    depth_time_accum_sec += round_elapsed
                    print(Fore.MAGENTA + f"[PBP] Round {rounds} time: {round_elapsed:.6f}s")
                    round_status = "wait_for_new_instance"
                    top_pas_yes_obj_id = None
                    llm_yesno_for_single_yes_group = None
                    if self.enable_pbp_refinement:
                        pas_by_obj = {int(item["object_id"]): float(item["pas"]) for item in refinement_pas}
                    else:
                        pas_by_obj = self._pas_for_non_refinement_text_goal(
                            nli_only_model=nli_only_model,
                            nli_only_tokenizer=nli_only_tokenizer,
                            candidates=candidates,
                            current_object_ids=current_object_ids,
                            category=category,
                            prop=prop,
                        )

                    if len(yes_object_ids) > 0 and len(no_object_ids) == 0:
                        top_pas_yes_obj_id = int(
                            max(
                                yes_object_ids,
                                key=lambda obj_id: (pas_by_obj.get(int(obj_id), float("-inf")), -int(obj_id)),
                            )
                        )
                        top_pas_yes_caption = str(candidates[top_pas_yes_obj_id].get("description", ""))
                        llm_yesno_for_single_yes_group = bool(
                            self._text_goal_candidate_yesno(
                                text_goal=text_goal,
                                category=category,
                                candidate_caption=top_pas_yes_caption,
                                candidate_prop=prop,
                            )
                        )
                        if llm_yesno_for_single_yes_group:
                            round_status = "single_yes_group_llm_yes"
                            current_object_ids = [top_pas_yes_obj_id]

                    logs.append(
                        {
                            **log_item,
                            "yes_object_ids": yes_object_ids,
                            "no_object_ids": no_object_ids,
                            "top_pas_yes_obj_id": top_pas_yes_obj_id,
                            "llm_yesno_for_single_yes_group": llm_yesno_for_single_yes_group,
                            "next_size": len(yes_object_ids) + len(no_object_ids),
                            "round_elapsed_time_sec": round_elapsed,
                            "pbp_round_status": round_status,
                        }
                    )
                    if round_status == "wait_for_new_instance":
                        if int(current_step) >= int(trigger_step):
                            if len(yes_object_ids) > 0:
                                fallback_pool = yes_object_ids
                            elif len(no_object_ids) > 0:
                                fallback_pool = no_object_ids
                            else:
                                fallback_pool = current_object_ids
                            forced_obj_id = int(
                                max(
                                    fallback_pool,
                                    key=lambda obj_id: (pas_by_obj.get(int(obj_id), float("-inf")), -int(obj_id)),
                                )
                            )
                            current_object_ids = [forced_obj_id]
                            logs[-1]["pbp_round_status"] = "forced_pas_fallback_trigger_step"
                            logs[-1]["forced_selected_object_id"] = forced_obj_id
                            print(Fore.YELLOW + f"[PBP][text_goal] Trigger reached (step={current_step}>= {trigger_step}). Force-selecting highest-PAS obj_id={forced_obj_id}.")
                        else:
                            forced_status = "PBP_WAIT_FOR_NEW_INSTANCE"
                            print(Fore.YELLOW + "[PBP][text_goal] Waiting for new matured instance.")
                        break
                    if round_status == "single_yes_group_llm_yes":
                        print(Fore.GREEN + f"[PBP][text_goal] Single-group case accepted by LLM. Selecting obj_id={top_pas_yes_obj_id}.")
                        break
                    continue

                prop_key = prop.strip().lower()
                prop_counts[prop_key] = int(prop_counts.get(prop_key, 0)) + 1
                if prop_counts[prop_key] >= int(self.max_same_property_streak):
                    logs.append(
                        {
                            **log_item,
                            "round_elapsed_time_sec": float(time.perf_counter() - round_start),
                            "pbp_round_status": "fallback_same_property_count",
                            "same_property_count": int(prop_counts[prop_key]),
                        }
                    )
                    print(Fore.RED + f"[PBP] Max same property streak {prop_counts[prop_key]} reached for '{prop}'. Finishing PBP.")
                    break
            else:
                print(Fore.YELLOW + "[PBP] Using LLM to propose binary property...")
                prompt = build_property_prompt(descriptions, used_properties, category)
                llm_out = self.llm_client.ask(prompt=prompt)
                prop = parse_property(llm_out)

                if not prop:
                    prop = self._mllm_find_property(candidates, current_object_ids, category, used_properties)
                    used_mllm_fallback = True

                if not prop:
                    round_elapsed = float(time.perf_counter() - round_start)
                    round_times_sec.append(round_elapsed)
                    depth_time_accum_sec += round_elapsed
                    print(Fore.MAGENTA + f"[PBP] Round {rounds} time: {round_elapsed:.6f}s")
                    logs.append(
                        {
                            "round": rounds,
                            "depth": curr_depth,
                            "note": "failed_to_find_prop",
                            "round_elapsed_time_sec": round_elapsed,
                            "pbp_round_status": "no_property_found",
                        }
                    )
                    continue

                log_item = {"round": rounds, "depth": curr_depth, "chosen_property": prop}

                prop_key = prop.strip().lower()
                prop_counts[prop_key] = int(prop_counts.get(prop_key, 0)) + 1
                if prop_counts[prop_key] >= int(self.max_same_property_streak):
                    logs.append(
                        {
                            **log_item,
                            "round_elapsed_time_sec": float(time.perf_counter() - round_start),
                            "pbp_round_status": "fallback_same_property_count",
                            "same_property_count": int(prop_counts[prop_key]),
                        }
                    )
                    print(Fore.RED + f"[PBP] Max same property streak {prop_counts[prop_key]} reached for '{prop}'. Finishing PBP.")
                    break

                # === Group Candidates by Property ===
                yes_object_ids, no_object_ids = self._group_candidates_by_property(
                    candidates, current_object_ids, prop, category
                )
                
                # Guard: degenerate split
                if len(yes_object_ids) == 0 or len(no_object_ids) == 0:
                    print(Fore.RED + f"[PBP] No effective partitioning with LLM. Retrying partitioning with MLLM.\nNon-effective property: '{prop}'")
                    if prop not in used_properties:
                        used_properties.append(prop)
                    if not used_mllm_fallback:
                        mllm_prop = self._mllm_find_property(candidates, current_object_ids, category, used_properties)
                        if mllm_prop:
                            prop = mllm_prop
                            used_mllm_fallback = True
                            log_item = {"round": rounds, "depth": curr_depth, "chosen_property": prop}

                            prop_key = prop.strip().lower()
                            prop_counts[prop_key] = int(prop_counts.get(prop_key, 0)) + 1
                            if prop_counts[prop_key] >= int(self.max_same_property_streak):
                                logs.append(
                                    {
                                        **log_item,
                                        "round_elapsed_time_sec": float(time.perf_counter() - round_start),
                                        "pbp_round_status": "fallback_same_property_count",
                                        "same_property_count": int(prop_counts[prop_key]),
                                    }
                                )
                                print(Fore.RED + f"[PBP] Max same property streak {prop_counts[prop_key]} reached for '{prop}'. Finishing PBP.")
                                break

                            yes_object_ids, no_object_ids = self._group_candidates_by_property(
                                candidates, current_object_ids, prop, category
                            )

                    if len(yes_object_ids) == 0 or len(no_object_ids) == 0:
                        print(Fore.RED + f"[PBP] No effective partitioning with MLLM. Retrying partitioning with LLM.\nNon-effective property: '{prop}'")
                        if prop not in used_properties:
                            used_properties.append(prop)
                        round_elapsed = float(time.perf_counter() - round_start)
                        round_times_sec.append(round_elapsed)
                        depth_time_accum_sec += round_elapsed
                        print(Fore.MAGENTA + f"[PBP] Round {rounds} time: {round_elapsed:.6f}s")
                        logs.append(
                            {
                                **log_item,
                                "yes_object_ids": yes_object_ids,
                                "no_object_ids": no_object_ids,
                                "next_size": len(yes_object_ids) + len(no_object_ids),
                                "round_elapsed_time_sec": round_elapsed,
                                "pbp_round_status": "degenerate_split",
                            }
                        )
                        continue
            
            if task_type == "text_goal":
                print(Fore.MAGENTA + f"[PBP][ROUND {rounds}] Interaction(text_goal): auto-select refined cohesive set (G_c').")
                target_has = True
            else:
                nq_before = int(self.vlm_oracle.ask_to_human_episode_counter.get(ep_id, 0))
                if nq_before >= int(coin_question_cap):
                    dialogue_history = self.vlm_oracle.pbp_dialogue_history.get(ep_id, [])
                    last_answer_yes = True
                    if len(dialogue_history) > 0:
                        last_answer_yes = str(dialogue_history[-1][1]).strip().lower() == "yes"
                    pas_by_obj = {int(item["object_id"]): float(item["pas"]) for item in refinement_pas}
                    fallback_pool = yes_object_ids if last_answer_yes else no_object_ids
                    if len(fallback_pool) == 0:
                        fallback_pool = current_object_ids
                    if last_answer_yes:
                        forced_obj_id = int(
                            max(
                                fallback_pool,
                                key=lambda obj_id: (pas_by_obj.get(int(obj_id), float("-inf")), -int(obj_id)),
                            )
                        )
                    else:
                        forced_obj_id = int(
                            min(
                                fallback_pool,
                                key=lambda obj_id: (pas_by_obj.get(int(obj_id), float("inf")), int(obj_id)),
                            )
                        )
                    round_elapsed = float(time.perf_counter() - round_start)
                    round_times_sec.append(round_elapsed)
                    depth_time_accum_sec += round_elapsed
                    print(Fore.MAGENTA + f"[PBP] Round {rounds} time: {round_elapsed:.6f}s")
                    logs.append(
                        {
                            **log_item,
                            "user_answer": bool(last_answer_yes),
                            "prev_object_ids": current_object_ids.copy(),
                            "current_object_ids": [forced_obj_id],
                            "candidate_count_decreased": len([forced_obj_id]) < len(current_object_ids),
                            "yes_object_ids": yes_object_ids,
                            "no_object_ids": no_object_ids,
                            "next_size": 1,
                            "round_elapsed_time_sec": round_elapsed,
                            "pbp_round_status": "forced_pas_fallback_question_cap",
                            "forced_selected_object_id": forced_obj_id,
                        }
                    )
                    current_object_ids = [forced_obj_id]
                    print(Fore.YELLOW + f"[PBP][coin] Question cap reached (NQ={nq_before}). Force-selecting obj_id={forced_obj_id}.")
                    break
                # === User Simulator ===
                target_has = self._user_simulator_has_property(
                    target_image, category, prop, ep_id
                )

                print(f"[PBP] Property: {prop} | User simulator answer:{Fore.GREEN if target_has else Fore.RED}{'Yes' if target_has else 'No'}{Fore.RESET}")
                
                # Store in dialogue history (VLMOracle)
                if ep_id not in self.vlm_oracle.pbp_dialogue_history:
                    self.vlm_oracle.pbp_dialogue_history[ep_id] = []
                self.vlm_oracle.pbp_dialogue_history[ep_id].append(
                    (prop, "Yes" if target_has else "No", curr_depth)
                )
                if task_type == "coin" and (not target_has) and len(no_object_ids) == 0:
                    if prop not in used_properties:
                        used_properties.append(prop)
                    round_elapsed = float(time.perf_counter() - round_start)
                    round_times_sec.append(round_elapsed)
                    depth_time_accum_sec += round_elapsed
                    print(Fore.MAGENTA + f"[PBP] Round {rounds} time: {round_elapsed:.6f}s")
                    nq_now = int(self.vlm_oracle.ask_to_human_episode_counter.get(ep_id, 0))
                    logs.append(
                        {
                            **log_item,
                            "user_answer": False,
                            "prev_object_ids": current_object_ids.copy(),
                            "current_object_ids": [],
                            "candidate_count_decreased": True,
                            "yes_object_ids": yes_object_ids,
                            "no_object_ids": no_object_ids,
                            "next_size": 0,
                            "round_elapsed_time_sec": round_elapsed,
                            "pbp_round_status": "wait_for_new_instance",
                        }
                    )
                    if int(current_step) >= int(trigger_step) or nq_now >= int(coin_question_cap):
                        pas_by_obj = {int(item["object_id"]): float(item["pas"]) for item in refinement_pas}
                        fallback_pool = no_object_ids if len(no_object_ids) > 0 else current_object_ids
                        forced_obj_id = int(
                            min(
                                fallback_pool,
                                key=lambda obj_id: (pas_by_obj.get(int(obj_id), float("inf")), int(obj_id)),
                            )
                        )
                        current_object_ids = [forced_obj_id]
                        logs[-1]["pbp_round_status"] = "forced_pas_fallback_trigger_or_question_cap"
                        logs[-1]["forced_selected_object_id"] = forced_obj_id
                        print(
                            Fore.YELLOW
                            + f"[PBP][coin] Forced fallback (step={current_step}, trigger_step={trigger_step}, NQ={nq_now}, cap={coin_question_cap}). "
                            + f"Selecting lowest-PAS obj_id={forced_obj_id}."
                        )
                    else:
                        forced_status = "PBP_WAIT_FOR_NEW_INSTANCE"
                        print(Fore.YELLOW + "[PBP][coin] User answered No but refined Gr is empty. Waiting for new matured instance.")
                    break
            # Move to next node
            prev_object_ids = current_object_ids.copy()
            current_object_ids = yes_object_ids if target_has else no_object_ids
            if prop not in used_properties:
                used_properties.append(prop)
            
            if task_type == "text_goal":
                pbp_round_status = "split_success_text_goal_auto_yes"
            else:
                pbp_round_status = "stuck_llm_fallback_to_mllm" if used_mllm_fallback else "split_success"

            round_elapsed = float(time.perf_counter() - round_start)
            round_times_sec.append(round_elapsed)
            depth_time_accum_sec += round_elapsed
            print(Fore.MAGENTA + f"[PBP] Round {rounds} time: {round_elapsed:.6f}s")
            logs.append({
                **log_item,
                "user_answer": bool(target_has),
                "prev_object_ids": prev_object_ids,
                "current_object_ids": current_object_ids,
                "candidate_count_decreased": len(current_object_ids) < len(prev_object_ids),
                "yes_object_ids": yes_object_ids,
                "no_object_ids": no_object_ids,
                "next_size": len(current_object_ids),
                "round_elapsed_time_sec": round_elapsed,
                "pbp_round_status": pbp_round_status
            })
            depth += 1
            depth_elapsed = float(depth_time_accum_sec)
            depth_times_sec.append(depth_elapsed)
            depth_avg_so_far = float(sum(depth_times_sec) / len(depth_times_sec))
            print(
                Fore.MAGENTA
                + f"[PBP] Depth {depth} time: {depth_elapsed:.6f}s (avg={depth_avg_so_far:.6f}s, n={len(depth_times_sec)})"
            )
            depth_time_accum_sec = 0.0
            
            print(Fore.GREEN + f"[PBP] Split success! Candidate change: {prev_object_ids} → {current_object_ids} {len(current_object_ids)} candidates remain.")
            if task_type == "text_goal":
                print(Fore.GREEN + f"[PBP] Auto decision(text_goal):{Fore.RESET} [{Fore.GREEN if target_has else Fore.RED}{'Yes' if target_has else 'No'}{Fore.RESET}]")
            else:
                print(Fore.GREEN + f"[PBP] User simulator answer:{Fore.RESET} [{Fore.GREEN if target_has else Fore.RED}{'Yes' if target_has else 'No'}{Fore.RESET}]")
            print(Fore.GREEN + f"[PBP] Effective property: [{prop}]")
            print("="*50)
        
        # === Finalization ===
        if forced_status is not None:
            selected_object_id, status, selected_success = None, forced_status, False
        else:
            selected_object_id, status, selected_success = self._finalize_selection(
                ep_id, category, target_image, candidates, current_object_ids,
                last_multi_candidate_node, used_properties, logs, depth, current_step, num_frontiers, task_type
            )
        
        elapsed_time = float(time.perf_counter() - start_time)
        round_time_total_sec = float(sum(round_times_sec))
        round_time_count = int(len(round_times_sec))
        round_avg_time_sec = round_time_total_sec / max(1, round_time_count)
        depth_time_total_sec = float(sum(depth_times_sec))
        depth_time_count = int(len(depth_times_sec))
        depth_avg_time_sec = depth_time_total_sec / max(1, depth_time_count)
        
        # Build result log
        dialogue_history = self.vlm_oracle.pbp_dialogue_history.get(ep_id, [])
        # Build candidate order for reproducibility
        candidate_order = [{"index": i+1, "object_id": obj_id} for i, obj_id in enumerate(sorted(candidates.keys()))]

        # Try to attach selected_index if available from fallback
        selected_index = getattr(self, '_last_selected_index', None)
        if hasattr(self, '_last_selected_index'):
            delattr(self, '_last_selected_index')

        pbp_run_log = {
            "pbp_rounds": rounds,
            "pbp_depth": depth,
            "used_properties": list(used_properties),
            "remaining_object_ids": [] if selected_success else current_object_ids,
            "yes_no_result": dialogue_history,  # List[(property, Yes/No, depth)]
            "elapsed_time": elapsed_time,
            "round_time_total_sec": round_time_total_sec,
            "round_time_count": round_time_count,
            "round_avg_time_sec": round_avg_time_sec,
            "depth_time_total_sec": depth_time_total_sec,
            "depth_time_count": depth_time_count,
            "depth_avg_time_sec": depth_avg_time_sec,
            "status": status,
            "fallback_reason": fallback_reason,
            "selected_object_id": selected_object_id,
            "selected_index": selected_index,
            "candidate_order": candidate_order,
            "logs": logs
        }
        
        print(Fore.CYAN + f"[PBP] Completed in {elapsed_time:.2f}s. Status: {status}")
        print(Fore.MAGENTA + f"[PBP] Round avg: {round_avg_time_sec:.6f}s (n={round_time_count})")
        print(Fore.MAGENTA + f"[PBP] Depth avg: {depth_avg_time_sec:.6f}s (n={depth_time_count})")
        print(Fore.CYAN + f"[PBP] Used properties: {used_properties}")
        print(Fore.CYAN + f"[PBP] Selected properties: {dialogue_history}")
        
        return selected_object_id, pbp_run_log
    
    def _mllm_find_property(
        self,
        all_candidates: Dict,
        current_object_ids: List[int],
        category: str,
        used_properties: List[str]
    ) -> Optional[str]:
        """Find property using MLLM from images."""
        images = []
        for obj_id in current_object_ids:
            diverse_views_image = all_candidates[obj_id]['diverse_views_image']
            images.append(diverse_views_image)
        
        if not images:
            return None
        
        prompt = build_mllm_property_prompt(category, len(images), used_properties)
        
        print(Fore.YELLOW + "[PBP] Calling MLLM to find property...")
        try:
            prop_text, _ = self.mllm_client.ask(
                images,
                prompt=prompt,
                return_token_likelihood=False,
            )
            prop_text = (prop_text or "").strip()
            prop_text = parse_property(prop_text)
            return prop_text
        except Exception as e:
            print(Fore.RED + f"[PBP] MLLM property extraction failed: {e}")
            return None
    
    def _group_candidates_by_property(
        self,
        all_candidates: Dict,
        current_object_ids: List[int],
        property_text: str,
        category: str
    ) -> Tuple[List[int], List[int]]:
        """
        Group candidates based on property using MLLM Yes/No.
        From tree_search.py group_from_scratch (Line 358-380).
        """
        yes_object_ids, no_object_ids = [], []
        
        for obj_id in current_object_ids:
            diverse_views_image = all_candidates[obj_id]['diverse_views_image']
            decision = self._mllm_yesno(diverse_views_image, category, property_text)
            
            if decision is False:
                no_object_ids.append(obj_id)
            else:  # True or None (fallback to yes)
                yes_object_ids.append(obj_id)
            print(f"MLLM decision for object/group ID {obj_id}: {Fore.GREEN if decision else Fore.RED}{'Yes' if decision else 'No'}{Fore.RESET}")
            print(f"property: [{property_text}]")
            print("="*50)
        
        return yes_object_ids, no_object_ids
    
    def _mllm_yesno(
        self,
        image: np.ndarray,
        category: str,
        property_text: str,
        use_user_simulator: bool = False,
        fallback_on_failure: Optional[bool] = True,
    ) -> Optional[bool]:
        """
        Ask MLLM if image has property (Yes/No).
        From tree_search.py mllm_yesno (Line 137-184).
        """
        prompt = build_yesno_property_prompt(category, property_text)
        
        self._last_mllm_completion_tokens = None
        try:
            # Use CoIN's wrapper
            mllm_client = self.user_simulator_mllm_client if use_user_simulator else self.mllm_client
            response_text, likelihood, completion_tokens = mllm_client.ask2(
                image=image,
                prompt=prompt,
                return_token_likelihood=True
            )
            if completion_tokens is not None:
                self._last_mllm_completion_tokens = int(completion_tokens)
            
            if likelihood is None:
                print(Fore.RED + "[PBP] No likelihood returned from MLLM")
                return fallback_on_failure
            
            decision = decide_from_likelihood(likelihood)
            return decision
            
        except Exception as e:
            print(Fore.RED + f"[PBP] MLLM Yes/No failed: {e}")
            return fallback_on_failure
    
    def _user_simulator_has_property(
        self,
        target_image: np.ndarray,
        category: str,
        property_text: str,
        ep_id: int,
        retries: int = 3
    ) -> bool:
        """
        Ask user simulator (target image) if it has property.
        Counts only valid responses (parseable Yes/No + completion token usage).
        """
        for attempt in range(retries + 1):
            # None means AskUser failure (e.g., response parsing failure, missing likelihood, API error/timeout).
            decision = self._mllm_yesno(
                target_image,
                category,
                property_text,
                use_user_simulator=True,
                fallback_on_failure=None,
            )
            completion_tokens = self._last_mllm_completion_tokens
            
            if decision is not None and completion_tokens is not None:
                self.vlm_oracle.ask_to_human_episode_counter[ep_id] = \
                    self.vlm_oracle.ask_to_human_episode_counter.get(ep_id, 0) + 1
                self.vlm_oracle.response_len_total_tokens_episode[ep_id] = \
                    self.vlm_oracle.response_len_total_tokens_episode.get(ep_id, 0) + int(completion_tokens)
                self.vlm_oracle.response_len_num_valid_responses_episode[ep_id] = \
                    self.vlm_oracle.response_len_num_valid_responses_episode.get(ep_id, 0) + 1
                print(Fore.BLUE + f"[PBP] User simulator decision: {Fore.GREEN if decision else Fore.RED}{'Yes' if decision else 'No'}{Fore.RESET}")
                print(f"property: [{property_text}]")
                print("="*50)
                return decision
            
            if attempt < retries:
                print(Fore.YELLOW + f"[PBP] AskUser failed. Retry {attempt + 1}")
        
        # Fallback
        print(Fore.RED + f"[PBP] AskUser failed during {retries} retries. Defaulting to 'yes' without counting a user question.")
        return True
    
    def _finalize_selection(
        self,
        ep_id: int,
        category: str,
        target_image: np.ndarray,
        candidates: Dict,
        current_object_ids: List[int],
        last_multi_candidate_node: Optional[List[int]],
        used_properties: List[str],
        logs: List[Dict],
        rounds: int,
        current_step: int,
        num_frontiers: int,
        task_type: str = "coin",
    ) -> Tuple[Optional[int], str, bool]:
        """
        Finalize object selection based on remaining candidates.
        
        Returns:
            (selected_object_id, status_string)
        """
        if len(current_object_ids) == 1:
            # Final verification
            candidate_obj_id = current_object_ids[0]
            candidate_obj = candidates[candidate_obj_id]
            if str(task_type).strip().lower() == "text_goal":
                is_match, new_prop = True, None
                print(Fore.CYAN + "[PBP] Final verification skipped for text_goal (no user simulator interaction).")
            else:
                is_match, new_prop = self._final_verification( # This is only for logging, not for selection. Doesn't affect outcome.
                    target_image,
                    candidate_obj.get('diverse_views_image'),
                    candidate_obj.get('description', ''),
                    used_properties,
                    ep_id,
                )
                print(Fore.CYAN + f"[PBP] Final verification result: {Fore.GREEN if is_match else Fore.RED}{'Match' if is_match else 'No Match'}{Fore.RESET}")
                print(f"Final verification property: [{new_prop}]")
            
            logs.append({
                "round": "final_verification",
                "depth": rounds,
                "is_match": is_match,
                "new_prop": new_prop,
                "pbp_round_status": "final_verification"
            })
            
            # Always select the single candidate regardless of match
            return candidate_obj_id, ("PBP_FALLBACK_SINGLE_CANDIDATE" if not is_match else "PBP_TARGET_FOUND"), bool(is_match)
        
        elif len(current_object_ids) >= 2:
            # Multiple candidates remain - run fallback selection
            selected_id = self._fallback_selection(
                ep_id, candidates, current_object_ids
            )
            return selected_id, "PBP_FALLBACK_MULTIPLE_CANDIDATES", False
        
        else:  # len == 0
            trigger_step = int(self.pbp_config.get("trigger_step", 400))
            if int(num_frontiers) > 0 and int(current_step) < int(trigger_step):
                print(
                    Fore.RED
                    + f"[PBP] No candidates remain before trigger_step={trigger_step} with frontiers left. Requesting trigger-threshold increase."
                )
                return None, "PBP_NO_CANDIDATES_INCREASE_TRIGGER_THRESHOLD", False
            # No candidates remain - use last multi-candidate node
            if last_multi_candidate_node and len(last_multi_candidate_node) >= 2:
                print(Fore.YELLOW + "[PBP] No candidates. Using last multi-candidate node.")
                selected_id = self._fallback_selection(
                    ep_id, candidates, last_multi_candidate_node
                )
                return selected_id, "PBP_FALLBACK_NO_CANDIDATES_REMAIN", False
            else:
                # No fallback possible
                return None, "PBP_FALLBACK_NO_CANDIDATES_REMAIN", False
    
    def _final_verification(
        self,
        target_image: np.ndarray,
        candidate_image: Optional[np.ndarray],
        candidate_description: str,
        used_properties: List[str],
        ep_id: int,
    ) -> Tuple[Optional[bool], Optional[str]]:
        """
        Final verification: compare target and candidate images.
        From tree_search.py mllm_final_verification (Line 261-319).
        """
        mode = getattr(self, "final_verification_mode", "candidate_text_mode")
        if mode not in {"candidate_image_mode", "candidate_text_mode"}:
            raise ValueError(f"[PBP] Invalid final_verification_mode: {mode}")
        if mode == "candidate_text_mode" and candidate_image is None:
            raise ValueError("[PBP] candidate_image is None but mode is not 'candidate_text_mode'.")
        
        prompt = build_final_verification_prompt(
            "object",
            used_properties,
            mode="candidate_text_mode" if mode == "candidate_text_mode" else "candidate_image_mode",
            candidate_description=candidate_description,
        )
        
        try:
            images = [target_image] if mode == "candidate_text_mode" else [target_image, candidate_image]
            output_text, _ = self.user_simulator_mllm_client.ask(
                images,
                prompt=prompt,
                return_token_likelihood=False,
            )
            output_text = (output_text or "").strip()
            
            # Parse decision (from tree_search.py Line 305)
            decision_match = re.search(r"DECISION:\s*(yes|no)", output_text, re.IGNORECASE)
            is_match = None
            if decision_match:
                is_match = decision_match.group(1).lower() == 'yes'
            
            # Parse property (from tree_search.py Line 313)
            new_property = None
            if is_match is False:
                reason_match = re.search(r"PROPERTY:\s*(.*)", output_text, re.IGNORECASE)
                if reason_match:
                    new_property = reason_match.group(1).strip()
                    if new_property.lower() == 'n/a':
                        new_property = None
            
            return is_match, new_property
            
        except Exception as e:
            print(Fore.RED + f"[PBP] Final verification failed: {e}")
            return None, None
    
    def _fallback_selection(
        self,
        ep_id: int,
        candidates: Dict,
        candidate_object_ids: List[int]
    ) -> int:
        """
        Select best candidate from multiple options using LLM.
        Uses dialogue history to guide selection.
        
        Returns:
            selected_object_id
        """
        # Build ordered mapping
        ordered_object_ids = sorted(candidate_object_ids)
        index_to_obj_id = {i+1: obj_id for i, obj_id in enumerate(ordered_object_ids)}
        
        # Get descriptions
        descriptions = [candidates[obj_id]['description'] for obj_id in ordered_object_ids]
        
        # Get dialogue history
        dialogue_history = self.vlm_oracle.pbp_dialogue_history.get(ep_id, [])
        
        # Build prompt
        prompt = build_fallback_selection_prompt(descriptions, dialogue_history)
        
        # Call LLM
        try:
            llm_out = self.llm_client.ask(prompt=prompt)
            
            # Parse SELECTED_INDEX (tree_search.py style)
            index_match = re.search(r"SELECTED_INDEX:\s*(\d+)", llm_out, re.IGNORECASE)
            
            if index_match:
                selected_index = int(index_match.group(1))
                
                # Validate range
                if 1 <= selected_index <= len(ordered_object_ids):
                    selected_obj_id = index_to_obj_id[selected_index]
                    print(Fore.GREEN + f"[PBP] Fallback selected: index={selected_index}, obj_id={selected_obj_id}")
                    # Store selected index for logging (caller will include in pbp_run_log)
                    self._last_selected_index = selected_index
                    return selected_obj_id
                else:
                    print(Fore.RED + f"[PBP] Invalid index {selected_index}. Using first candidate.")
            else:
                print(Fore.RED + "[PBP] Failed to parse SELECTED_INDEX. Using first candidate.")
            
        except Exception as e:
            print(Fore.RED + f"[PBP] Fallback selection failed: {e}. Using first candidate.")
        
        # Fallback: return first candidate
        return ordered_object_ids[0]
