
This repo contains code and instructions to analyze the emotional/mental state of a person in a short video clip using Qwen3-VL.

1. Python 3.12 is recommended
2. Install dependencies using the requirements.txt (custom pytorch installation may be necessary if your environment uses an older CUDA version)
3. Run `python inference.py -i video.mp4 `



### VRAM Requirements

30B-A3B:  60GB VRAM.  
8B:  18GB VRAM.  
2B:  5GB VRAM.  

Depending on the video and hyperparameters, an additional 5-30GB VRAM may be necessary.   


--- 


<details>

<summary> Preprocessing </summary>

## ffmpeg

**Crop**  
`ffmpeg -i file -vf "crop=w:h:x:y out.mp4`  
x,y -> top left coords  
w,h -> crop size 

**Preview**  
`ffplay -i file -vf "crop=w:h:x:y"`

</details>