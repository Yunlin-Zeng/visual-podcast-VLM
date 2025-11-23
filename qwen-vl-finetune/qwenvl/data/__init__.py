import re

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": f"PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",
    "data_path": f"",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}

# Custom dataset: Podcast transcript generation training data
PODCAST_QWEN_LORA = {
    "annotation_path": "/home/ubuntu/image-to-text/data/2025-10-16_prompt_templates_tune_qwen3_vl/qwen_training_data.json",
    "data_path": "",  # Empty because image paths are absolute in annotation
}

# SPoRC podcast excerpt dataset with speaker-labeled dialogue (894 complete excerpts)
SPORC_EXCERPT = {
    "annotation_path": "/home/ubuntu/image-to-text/Qwen3-VL/data/qwen_training_data_sporc_excerpt.json",
    "data_path": "",  # Empty because image paths are relative in annotation
}

# SPoRC with working prompt format (800 word target, Q&A structure with final instruction)
SPORC_EXCERPT_WORKING_PROMPT = {
    "annotation_path": "/home/ubuntu/image-to-text/Qwen3-VL/data/qwen_training_data_sporc_excerpt_working_prompt.json",
    "data_path": "",
}

# SPoRC with FULL DETAILED prompt from Nov 9 Test 1 (with examples, 600-700 words)
SPORC_EXCERPT_DETAILED_PROMPT = {
    "annotation_path": "/home/ubuntu/image-to-text/Qwen3-VL/data/qwen_training_data_sporc_detailed_prompt.json",
    "data_path": "",
}

# Single sample for overfitting test (ep1131_ex0, 673 words, working Prompt 2 format)
SPORC_OVERFIT_SINGLE = {
    "annotation_path": "/home/ubuntu/image-to-text/Qwen3-VL/data/qwen_overfit_test_single_sample.json",
    "data_path": "",
}

# SPoRC with 235B-aligned prompt (topics/themes focused, 800 words, Nov 11 2025)
SPORC_235B_ALIGNED = {
    "annotation_path": "/home/ubuntu/image-to-text/Qwen3-VL/data/qwen_training_data_sporc_235b_aligned.json",
    "data_path": "",
}

data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "cambrian_737k_pack": CAMBRIAN_737K_PACK,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
    "podcast_qwen_lora": PODCAST_QWEN_LORA,
    "sporc_excerpt": SPORC_EXCERPT,
    "sporc_excerpt_working_prompt": SPORC_EXCERPT_WORKING_PROMPT,
    "sporc_excerpt_detailed_prompt": SPORC_EXCERPT_DETAILED_PROMPT,
    "sporc_overfit_single": SPORC_OVERFIT_SINGLE,
    "sporc_235b_aligned": SPORC_235B_ALIGNED,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
