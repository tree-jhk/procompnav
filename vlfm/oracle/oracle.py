from PIL import Image
from vlfm.utils.prompts import (
    LLaVa_TARGET_OBJECT_IS_DETECTED,
    LLaVa_TARGET_OBJECT_IS_DETECTED_NEARBY,
)
from typing import List
import logging
from colorama import Fore
from colorama import init as _init_co
from retrying import retry

_init_co(autoreset=True)
import yaml
import numpy as np

from collections import defaultdict


class VLMOracle:

    def __init__(self, llava_client, llm_client) -> None:
        self.LMM_CLIENT = llava_client
        self.LLM_CLIENT = llm_client
        self.model_name = llava_client.model_name
        self.instance_image = None
        self.ep_id = None
        self.pbp_module = None
        self.instance_image_description = None  # description of the target image
        self.ask_to_human_episode_counter = {}
        self.response_len_total_tokens_episode = {}
        self.response_len_num_valid_responses_episode = {}
        self.pbp_dialogue_history = {}  # {ep_id: List[(property, Yes/No, depth)]}

        # this ensure that no LMM is used to answer question, and that LLM does not perform answer checking
        # useful to use it for API call
        self.HUMAN_HAS_TO_ANSWER_THE_QUESTION = False

        self.questions_for_target_object_list = defaultdict(list)
        self.answers_for_target_object_list = defaultdict(list)
        self.facts_for_target_object_list = defaultdict(list)

    def get_instance_image(self):
        return self.instance_image

    def set_instance_image(self, instance_image, target_object, task_type: str = "", text_goal: str = "", ep_id=None):
        self.instance_image = instance_image
        if ep_id is not None:
            self.ep_id = ep_id
        self.set_image_description(instance_image, target_object)
        if str(task_type).strip().lower() == "text_goal" and self.pbp_module is not None:
            category = str(target_object).split("|")[0]
            instruction = str(text_goal or "").strip()
            if not instruction:
                instruction = str(getattr(self.pbp_module, "_text_goal_property_source", "")).strip()
            ep = int(self.ep_id) if self.ep_id is not None else -1
            attrs = self.pbp_module._get_text_goal_property_candidates(
                ep_id=ep,
                text_goal=instruction,
                category=category,
            )
            print(Fore.MAGENTA + f"[TEXT_GOAL][EP {ep}] text_goal: {instruction}")
            print(Fore.MAGENTA + f"[TEXT_GOAL][EP {ep}] attributes: {attrs}")

    # max retry 10 times, wait 10 seconds between each retry
    @retry(stop_max_attempt_number=10, wait_fixed=10000)
    def set_image_description(self, image, target_object):
        prompt = LLaVa_TARGET_OBJECT_IS_DETECTED.format(target_object=target_object) + f"\nThe {target_object} is definitely present in the image. Just describe it."
        image_description, _ = self.LMM_CLIENT.ask(np.array(image), prompt=prompt)

        nearby_prompt = LLaVa_TARGET_OBJECT_IS_DETECTED_NEARBY.format(target_object=target_object)
        nearby_description, _ = self.LMM_CLIENT.ask(np.array(image), prompt=nearby_prompt)

        image_description += "\n\nRelation with nearby objects:\n" + nearby_description

        self.instance_image_description = image_description
        print(Fore.LIGHTCYAN_EX + f"Image description: {image_description}")
        # Image.fromarray(self.instance_image).save("instance_image.png")

    def get_description_of_the_image(self):
        return self.instance_image_description

    def reset(self):
        self.instance_image = None
        self.ep_id = None
        self.instance_image_description = None
        self.questions_for_target_object_list.clear()
        self.answers_for_target_object_list.clear()
        self.facts_for_target_object_list.clear()
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def how_many_question_to_the_user(self, ep_id):
        return self.ask_to_human_episode_counter.get(ep_id, 0)
