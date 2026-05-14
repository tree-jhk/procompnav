import os
import json
import sys
import shutil
import re
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
from transformers import AutoModel, AutoProcessor, AutoImageProcessor
from transformers.image_utils import load_image

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

from colorama import Fore
from colorama import init as init_colorama
from vlfm.utils.pbp_prompts import build_occluded_view_filter_prompt, build_text_goal_candidate_yesno_prompt

init_colorama()


def merge_instances_into_groups(episode_root, episode_goal_category=None, eps=1, min_samples=1):
    # episode_root = f"/workspace/CoIN/eval_folder/<date>/<episode_id>/"
    detected_objects_jsonl_path = os.path.join(episode_root, "instances.jsonl")

    if not os.path.exists(detected_objects_jsonl_path):
        return None
    with open(detected_objects_jsonl_path, "r") as f:
        detected_objects_data = [json.loads(line) for line in f.readlines()]
    if len(detected_objects_data) == 0:
        return None
    
    instance_key_to_position = {}
    for data in detected_objects_data:
        if episode_goal_category is None:
            episode_goal_category = data["object_name"]

        instance_key = data.get("instance_id")
        if not isinstance(instance_key, str) or not instance_key:
            legacy_instance_num = data.get("instance_num")
            if isinstance(legacy_instance_num, int):
                instance_key = f"{episode_goal_category}_{legacy_instance_num:03d}"
            elif isinstance(legacy_instance_num, str) and legacy_instance_num.isdigit():
                instance_key = f"{episode_goal_category}_{int(legacy_instance_num):03d}"
            else:
                continue

        center_point_cloud_current = data.get("instance_position")
        if not (isinstance(center_point_cloud_current, list) and len(center_point_cloud_current) == 2):
            center_point_cloud_current = data.get("2D_center_point_cloud")  # Backward compatibility.
        if not (isinstance(center_point_cloud_current, list) and len(center_point_cloud_current) == 2):
            continue

        estimated_position = np.array(center_point_cloud_current)
        instance_key_to_position[instance_key] = estimated_position

    dir_detected_objects = os.path.join(episode_root, "detected_objects")
    if not os.path.exists(dir_detected_objects):
        return None

    old_group_dirs = sorted(
        [
            d
            for d in os.listdir(dir_detected_objects)
            if d.startswith("group_") and d.endswith("_merged")
        ]
    )
    for old_group_dir in old_group_dirs:
        shutil.rmtree(os.path.join(dir_detected_objects, old_group_dir))

    singleton_result = {}
    for idx, instance_key in enumerate(sorted(instance_key_to_position.keys())):
        k = f"group_{idx:03d}"
        center = instance_key_to_position[instance_key]
        members = [instance_key]
        singleton_result[k] = {"rep": instance_key, "center": center, "members": members}

        dir_merged_images = os.path.join(dir_detected_objects, f"{k}_merged")
        os.makedirs(dir_merged_images, exist_ok=True)

        # copy png images from each member instance directory to the merged directory
        for member in members:
            instance_subdir = os.path.join(dir_detected_objects, member)
            if not os.path.exists(instance_subdir):
                continue
            image_files = [f for f in os.listdir(instance_subdir) if f.endswith(".png")]
            for img_file in image_files:
                src_path = os.path.join(instance_subdir, img_file)
                dst_path = os.path.join(dir_merged_images, img_file)
                shutil.copyfile(src_path, dst_path)

        # save group_info.json in the merged directory
        group_info = {
            "position_group_center": center.tolist(),
            "members": members,
            "position_of_each_member": {m: instance_key_to_position[m].tolist() for m in members}
        }
        group_info_file_path = os.path.join(dir_merged_images, "group_info.json")
        with open(group_info_file_path, "w") as f:
            json.dump(group_info, f, indent=2)

    return singleton_result


def merge_instances_into_groups_caption(
    episode_root,
    episode_goal_category,
    llm_connector,
    text_only_model,
    similarity_threshold=0.9,
):
    dir_episode_detected_objects = os.path.join(episode_root, "detected_objects")
    group_dir_names = sorted(
        [
            d
            for d in os.listdir(dir_episode_detected_objects)
            if d.startswith("group_") and d.endswith("_merged")
        ]
    )
    if len(group_dir_names) == 0:
        return {}

    group_records = []
    for group_dir_name in group_dir_names:
        group_path = os.path.join(dir_episode_detected_objects, group_dir_name)
        group_info_path = os.path.join(group_path, "group_info.json")
        with open(group_info_path, "r") as f:
            group_info = json.load(f)
        group_records.append(
            {
                "dir_name": group_dir_name,
                "dir_path": group_path,
                "group_info_path": group_info_path,
                "group_info": group_info,
                "caption": str(group_info.get("caption", "")),
            }
        )

    captions = [item["caption"] for item in group_records]
    text_embeddings = text_only_model.encode(captions, batch_size=16, convert_to_numpy=True)
    if not isinstance(text_embeddings, np.ndarray):
        text_embeddings = np.array(text_embeddings)
    text_embeddings = text_embeddings / (np.linalg.norm(text_embeddings, axis=1, keepdims=True) + 1e-12)
    similarity_matrix = np.matmul(text_embeddings, text_embeddings.T)

    n_groups = len(group_records)
    parents = list(range(n_groups))

    def _find(x):
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    def _union(a, b):
        root_a = _find(a)
        root_b = _find(b)
        if root_a == root_b:
            return
        if root_a < root_b:
            parents[root_b] = root_a
        else:
            parents[root_a] = root_b

    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            if float(similarity_matrix[i, j]) < float(similarity_threshold):
                continue
            yesno_prompt = build_text_goal_candidate_yesno_prompt(
                text_goal=captions[i],
                category=episode_goal_category,
                candidate_caption=captions[j],
            )
            yesno_output = str(llm_connector.ask(prompt=yesno_prompt)).strip().lower()
            yesno_tokens = re.findall(r"\b(yes|no)\b", yesno_output)
            if len(yesno_tokens) > 0 and yesno_tokens[0] == "yes":
                _union(i, j)

    components = {}
    for idx in range(n_groups):
        root_idx = _find(idx)
        if root_idx not in components:
            components[root_idx] = []
        components[root_idx].append(idx)

    for component_indices in components.values():
        if len(component_indices) <= 1:
            continue
        sorted_component_indices = sorted(component_indices, key=lambda i: group_records[i]["dir_name"])
        keep_idx = max(
            sorted_component_indices,
            key=lambda i: (
                len(group_records[i]["group_info"].get("diverse_view_image_filenames", []))
                if isinstance(group_records[i]["group_info"].get("diverse_view_image_filenames", []), list)
                else 0
            ),
        )
        keep_record = group_records[keep_idx]
        keep_dir_path = keep_record["dir_path"]
        keep_group_info = keep_record["group_info"]

        merged_members = []
        merged_position_of_each_member = {}
        for idx in sorted_component_indices:
            current_record = group_records[idx]
            current_group_info = current_record["group_info"]
            members = current_group_info.get("members", [])
            if isinstance(members, list):
                merged_members.extend([m for m in members if isinstance(m, str)])
            position_of_each_member = current_group_info.get("position_of_each_member", {})
            if isinstance(position_of_each_member, dict):
                for member_key, member_pos in position_of_each_member.items():
                    if isinstance(member_key, str) and isinstance(member_pos, list) and len(member_pos) == 2:
                        merged_position_of_each_member[member_key] = [float(member_pos[0]), float(member_pos[1])]

            if idx == keep_idx:
                continue
            src_dir_path = current_record["dir_path"]
            for file_name in os.listdir(src_dir_path):
                if file_name == "group_info.json":
                    continue
                src_path = os.path.join(src_dir_path, file_name)
                dst_path = os.path.join(keep_dir_path, file_name)
                if os.path.exists(dst_path):
                    file_stem, file_ext = os.path.splitext(file_name)
                    dst_path = os.path.join(
                        keep_dir_path,
                        f"{file_stem}_{current_record['dir_name']}{file_ext}",
                    )
                shutil.move(src_path, dst_path)
            shutil.rmtree(src_dir_path)

        merged_members = sorted(set(merged_members))
        keep_group_info["members"] = merged_members
        keep_group_info["position_of_each_member"] = merged_position_of_each_member
        if len(merged_position_of_each_member) > 0:
            merged_positions = np.array(list(merged_position_of_each_member.values()), dtype=np.float32)
            keep_group_info["position_group_center"] = merged_positions.mean(axis=0).tolist()
        with open(keep_record["group_info_path"], "w") as f:
            json.dump(keep_group_info, f, indent=2)

    final_group_dir_names_before_reindex = sorted(
        [
            d
            for d in os.listdir(dir_episode_detected_objects)
            if d.startswith("group_") and d.endswith("_merged")
        ]
    )
    temp_group_dir_names = []
    for idx, group_dir_name in enumerate(final_group_dir_names_before_reindex):
        src_dir_path = os.path.join(dir_episode_detected_objects, group_dir_name)
        temp_dir_name = f"group_tmp_{idx:03d}_merged"
        temp_dir_path = os.path.join(dir_episode_detected_objects, temp_dir_name)
        os.rename(src_dir_path, temp_dir_path)
        temp_group_dir_names.append(temp_dir_name)
    for idx, temp_dir_name in enumerate(temp_group_dir_names):
        temp_dir_path = os.path.join(dir_episode_detected_objects, temp_dir_name)
        final_dir_name = f"group_{idx:03d}_merged"
        final_dir_path = os.path.join(dir_episode_detected_objects, final_dir_name)
        os.rename(temp_dir_path, final_dir_path)

    final_group_dir_names = sorted(
        [
            d
            for d in os.listdir(dir_episode_detected_objects)
            if d.startswith("group_") and d.endswith("_merged")
        ]
    )
    final_group_infos = {}
    for group_dir_name in final_group_dir_names:
        group_info_path = os.path.join(dir_episode_detected_objects, group_dir_name, "group_info.json")
        with open(group_info_path, "r") as f:
            final_group_infos[group_dir_name] = json.load(f)
    return final_group_infos




def make_figure_canvas(images, selected_indices, default_num_views=4):
    # if len(selected_indices) < default_num_views
    if len(selected_indices) < default_num_views:
        # make a canvas with 1 row and len(selected_indices) columns, and then plot images
        fig, axes = plt.subplots(1, len(selected_indices), figsize=(3 * len(selected_indices), 3))

        if len(selected_indices) == 1:
            axes = [axes]
            # plot one image
            sel_idx = selected_indices[0]
            axes[0].imshow(images[sel_idx])
            axes[0].set_title(f"View 0", fontsize=12, weight='bold')
            axes[0].axis('off')
        else:
            for i, ax in enumerate(axes):
                sel_idx = selected_indices[i]
                ax.imshow(images[sel_idx])
                
                ax.set_title(f"View {i}", fontsize=12, weight='bold')
                ax.axis('off')
        return fig, axes
    
    # if len(selected_indices) >= default_num_views
    if default_num_views == 4:
        fig, axes = plt.subplots(2, 2, figsize=(6, 6))
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            sel_idx = selected_indices[i]
            ax.imshow(images[sel_idx])
            
            ax.set_title(f"View {i}", fontsize=12, weight='bold')
            ax.axis('off')
        return fig, axes
    elif default_num_views == 6:
        fig, axes = plt.subplots(2, 3, figsize=(9, 6))
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            sel_idx = selected_indices[i]
            ax.imshow(images[sel_idx])
            
            ax.set_title(f"View {i}", fontsize=12, weight='bold')
            ax.axis('off')
        return fig, axes
    else:
        return None, None



def build_representative_view_selection_prompt(episode_goal_category, n_views):
    return (
        f"You are given one image composed of {n_views} views of a [{episode_goal_category}].\n"
        f"Views are labeled as View 0 to View {n_views - 1}.\n"
        f"Choose the single view where the [{episode_goal_category}] is shown most clearly.\n"
        "Answer only in this format:\n"
        "SELECTED_VIEW: <index>"
    )


def representative_view_selection(vlm_connector, diverse_views_image_path, episode_goal_category, diverse_view_image_filenames):
    n_views = len(diverse_view_image_filenames)
    representative_view_index = 0
    if n_views == 1:
        representative_view_index = 0
    elif n_views >= 2:
        select_prompt = build_representative_view_selection_prompt(episode_goal_category, n_views)
        selected_view_text, _ = vlm_connector.ask(image=diverse_views_image_path, prompt=select_prompt, return_token_likelihood=False)
        selected_view_matches = re.findall(r"SELECTED_VIEW:\s*(\d+)", selected_view_text, flags=re.IGNORECASE)
        if len(selected_view_matches) == 1:
            parsed_index = int(selected_view_matches[0])
            if 0 <= parsed_index < n_views:
                representative_view_index = parsed_index
            else:
                print(Fore.RED + f"[MULTI_VIEW][MLLM_ERROR] Invalid SELECTED_VIEW index: {parsed_index}. Fallback to View 0.")
        else:
            print(Fore.RED + f"[MULTI_VIEW][MLLM_ERROR] Failed to parse SELECTED_VIEW from: {selected_view_text}. Fallback to View 0.")
    else:
        print(Fore.RED + "[MULTI_VIEW][MLLM_ERROR] No view list found in group_info.json. Fallback to View 0.")
    representative_view_filename = None if n_views == 0 else diverse_view_image_filenames[representative_view_index]
    return representative_view_index, representative_view_filename


def select_occluded_panel_indices(vlm_connector, diverse_views_image_path, episode_goal_category, n_views):
    filter_prompt = build_occluded_view_filter_prompt(episode_goal_category, n_views)
    filtered_view_text, _ = vlm_connector.ask(image=diverse_views_image_path, prompt=filter_prompt, return_token_likelihood=False)
    filtered_lines = [line.strip() for line in filtered_view_text.splitlines() if line.strip()]
    remove_panel_indices = []
    for line in filtered_lines:
        if not line.isdigit():
            print(Fore.RED + f"[MULTI_VIEW][MLLM_ERROR] Failed to parse occluded-view indices from: {filtered_view_text}. Fallback to original KMeans-selected views.")
            return None
        parsed_idx = int(line)
        if parsed_idx < 0 or parsed_idx >= n_views:
            print(Fore.RED + f"[MULTI_VIEW][MLLM_ERROR] Failed to parse occluded-view indices from: {filtered_view_text}. Fallback to original KMeans-selected views.")
            return None
        if parsed_idx not in remove_panel_indices:
            remove_panel_indices.append(parsed_idx)
    return remove_panel_indices


def select_diverse_view_for_each_group(episode_root,
                                       episode_goal_category=None,
                                       image_embedding_model="facebook/dinov2-large",
                                       cuda_device_id=1,
                                       desired_num_clusters=6,
                                       default_num_views=6,
                                       vlm_connector=None,
                                       image_only_model=None,
                                       image_only_processor=None,
                                       ):
    if image_only_model is None or image_only_processor is None:
        model = AutoModel.from_pretrained(image_embedding_model, device_map=f"cuda:{cuda_device_id}", dtype=torch.bfloat16).eval()
        processor = AutoImageProcessor.from_pretrained(image_embedding_model)
    else:
        model = image_only_model
        processor = image_only_processor

    if episode_goal_category is None:
        with open(os.path.join(episode_root, "info.json"), "r") as f:
            episode_info = json.load(f)
            episode_goal_category = episode_info["object_category"]
    
    dir_episode_detected_objects = os.path.join(episode_root, "detected_objects")
    dir_merged_instance_groups = [os.path.join(dir_episode_detected_objects, d) for d in os.listdir(dir_episode_detected_objects) if d.startswith("group_")]

    for dir_group in dir_merged_instance_groups:
        image_files = sorted([f for f in os.listdir(dir_group) if f.endswith('.png')])
        image_files = [f for f in image_files if f.startswith(episode_goal_category.replace(" ", "_"))]
        images = [load_image(os.path.join(dir_group, f)) for f in image_files]

        inputs = processor(images=images, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden_states = outputs[0]

        img_features = last_hidden_states[:, 1:, :]  # torch.Size([N, 256, 1024])
        img_features = img_features.reshape(img_features.shape[0], -1)
        img_features = img_features.float().cpu().numpy() # (number of images, feature_dim1*feature_dim2)
        img_features = img_features / (np.linalg.norm(img_features, axis=1, keepdims=True) + 1e-12)

        if len(images) < desired_num_clusters:
            n_clusters = len(images)
        else:
            n_clusters = desired_num_clusters
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(img_features)

        # sometimes, some clusters may have no assigned images
        # then, we should set n_clusters to the actual number of clusters found
        unique_labels = set(cluster_labels)
        n_clusters = len(unique_labels)

        cluster_representative_image_idx = {} # {cluster_id: image_index}

        for c_id in range(n_clusters):
            indices_in_cluster = np.where(cluster_labels == c_id)[0]
            
            features_in_cluster = img_features[indices_in_cluster]
            centroid = np.mean(features_in_cluster, axis=0).reshape(1, -1)

            distances_to_centroid = np.linalg.norm(features_in_cluster - centroid, axis=1)
            closest_local_idx = np.argmin(distances_to_centroid)
            representative_idx = indices_in_cluster[closest_local_idx]

            cluster_representative_image_idx[c_id] = representative_idx

        selected_indices = list(cluster_representative_image_idx.values())
        selected_indices = selected_indices[:min(len(selected_indices), default_num_views)]
        diverse_view_image_filenames = [image_files[i] for i in selected_indices]
        save_filename = os.path.join(dir_group, 'diverse_views_dino.png')

        fig, axes = make_figure_canvas(images, selected_indices, default_num_views=default_num_views)
        fig.tight_layout()
        fig.savefig(save_filename, dpi=300, bbox_inches='tight')
        plt.close()

        if vlm_connector is not None and len(selected_indices) > 1:
            remove_panel_indices = select_occluded_panel_indices(
                vlm_connector,
                save_filename,
                episode_goal_category,
                len(selected_indices),
            )
            if remove_panel_indices is not None:
                remove_set = set(remove_panel_indices)
                keep_panel_indices = [i for i in range(len(selected_indices)) if i not in remove_set]
                if len(keep_panel_indices) > 0:
                    filtered_selected_indices = [selected_indices[i] for i in keep_panel_indices]
                    if len(filtered_selected_indices) != len(selected_indices):
                        selected_indices = filtered_selected_indices
                        diverse_view_image_filenames = [image_files[i] for i in selected_indices]
                        fig, axes = make_figure_canvas(images, selected_indices, default_num_views=default_num_views)
                        fig.tight_layout()
                        fig.savefig(save_filename, dpi=300, bbox_inches='tight')
                        plt.close()
                else:
                    print(Fore.RED + "[MULTI_VIEW][MLLM_ERROR] All views were removed by filter. Fallback to original KMeans-selected views.")

        group_info_file_path = os.path.join(dir_group, "group_info.json")
        group_info = {}
        if os.path.exists(group_info_file_path):
            with open(group_info_file_path, "r") as f:
                group_info = json.load(f)
        group_info["diverse_view_image_filenames"] = diverse_view_image_filenames
        with open(group_info_file_path, "w") as f:
            json.dump(group_info, f, indent=2)


def generate_caption_for_each_group(
    vlm_connector,
    episode_root,
    episode_goal_category=None
):
    if episode_goal_category is None:
        with open(os.path.join(episode_root, "info.json"), "r") as f:
            episode_info = json.load(f)
            episode_goal_category = episode_info["object_category"]

    dir_episode_detected_objects = os.path.join(episode_root, "detected_objects")
    dir_merged_instance_groups = [os.path.join(dir_episode_detected_objects, d) for d in os.listdir(dir_episode_detected_objects) if d.startswith("group_")]

    # in each group directory, there is a diverse_views_dino.png file
    # call vlm_connector.ask(image = image_path, prompt = prompt, return_token_likelihood=False)

    print(Fore.MAGENTA + f"[MULTI_VIEW] Generating captions for each group. num of groups: {len(dir_merged_instance_groups)}, goal category: {episode_goal_category}")
    for idx, dir_group in enumerate(dir_merged_instance_groups):
        print(f"Processing group directory: {idx+1} / {len(dir_merged_instance_groups)}")
        diverse_views_image_path = os.path.join(dir_group, 'diverse_views_dino.png')
        if not os.path.exists(diverse_views_image_path):
            continue

        group_info_file_path = os.path.join(dir_group, "group_info.json")
        group_info = {}
        if os.path.exists(group_info_file_path):
            with open(group_info_file_path, "r") as f:
                group_info = json.load(f)
        diverse_view_image_filenames = group_info.get("diverse_view_image_filenames", [])
        representative_view_index, representative_view_filename = representative_view_selection(
            vlm_connector,
            diverse_views_image_path,
            episode_goal_category,
            diverse_view_image_filenames,
        )
        if representative_view_filename is not None:
            print(Fore.GREEN + f"[MULTI_VIEW] Representative view selected. panel_index={representative_view_index}, view_filename={representative_view_filename}")
        group_info["representative_view_index"] = representative_view_index
        group_info["representative_view_filename"] = representative_view_filename
        
        prompt = f"""You are given an image which shows a {episode_goal_category} from multiple viewpoints. You need to describe the {episode_goal_category}, by combining the information from these different viewpoints.
First, describe the {episode_goal_category} in detail, focusing on its appearance and distinctive features (use only: color, shape).
Then, describe the other objects (use only: color, shape) close to the {episode_goal_category}, and their spatial relationships (use only: ['next to', 'on top of', 'under']) relative to the {episode_goal_category}.
Mention all clearly visible nearby objects around the {episode_goal_category}.
"""
        caption, _ = vlm_connector.ask(image=diverse_views_image_path, prompt=prompt, return_token_likelihood=False)
        print(Fore.MAGENTA + f"[MULTI_VIEW] Generated caption for group {idx+1} / {len(dir_merged_instance_groups)}: {caption}")

        # save caption into group_info.json in the group directory
        group_info["caption"] = caption
        with open(group_info_file_path, "w") as f:
            json.dump(group_info, f, indent=2)
