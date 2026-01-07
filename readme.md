
This repo contains code and instructions to analyze the emotional/mental state of a person in a short video clip using Qwen3-VL.

1. Python 3.12 is recommended
2. Install dependencies using the requirements.txt (custom pytorch installation may be necessary if your environment uses an older CUDA version)
3. Run `python inference.py`



### VRAM Requirements

For the Qwen3-VL-30B-A3B (MoE) version, the model itself requires 60GB VRAM, with another 30GB with video.
For the Qwen3-VL-8B version, the model itself requires .. VRAM.

Depending on the video and hyperparameters, an additional 10GB VRAM should be available.   


--- 


<details open>

<summary> Preprocessing </summary>

## ffmpeg

**Crop**  
`ffmpeg -i file -vf "crop=w:h:x:y out.mp4`  
x,y -> top left coords  
w,h -> crop size 

**Preview**  
`ffplay -i file -vf "crop=w:h:x:y"`


</details>