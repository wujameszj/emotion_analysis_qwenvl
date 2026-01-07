from time import time
from pathlib import Path
from argparse import ArgumentParser

from transformers import AutoModelForImageTextToText, AutoProcessor
from torch import bfloat16
from qwen_vl_utils import process_vision_info


video_param = {
    "type": "video",
    "video": "video.mp4",
    "min_pixels": 4 * 32 * 32,
    "max_pixels": 256 * 32 * 32,
    "total_pixels": 20480 * 32 * 32,
}

text_param = {
    "type": "text",
    "text": "Describe the person's emotional or mental state or changes therein."
}


def run(messages: str):

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, video_kwargs = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)

    if videos is not None:
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)
    else:
        video_metadatas = None

    # since qwen-vl-utils has resize the images/videos, we should pass do_resize=False to avoid duplicate operation in processor
    inputs = processor(text=text, images=images, videos=videos, video_metadata=video_metadatas, return_tensors="pt", do_resize=False, **video_kwargs)
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0] 


if __name__ == "__main__":
    ap = ArgumentParser("This program loads Qwen3-VL to analyze emotions in a video.")
    ap.add_argument("-i", "--input", type=str)
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--prompt", type=str, default="Describe the person's emotional or mental state or changes therein.")
    ap.add_argument("--config", type=str, default="")
    ap.add_argument("--out_path", type=str, default="out.txt")
    args = ap.parse_args()
    
    t = time()
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=bfloat16, 
        attn_implementation="flash_attention_2", 
        device_map="auto")
    
    print("Model loaded in", int(time()-t), "seconds");  t = time()

    input_path = Path(args.input)
    if input_path.is_file():
        video_param["video"] = args.input
        text_param["text"] = args.prompt
        msg = {
            "role": "user",
            "content": [video_param, text_param]
        }
        print(run([msg]))
        print(int(time()-t), "seconds elapsed.")
