
This repo contains code and instructions to analyze the emotional/mental state of a person in a short video clip using Qwen3-VL.

1. Python 3.12 is tested to work.
2. Install [pytorch](https://pytorch.org/get-started/locally) according to your system/CUDA version.
3. `pip install requirements.txt`
4. Optional: install flash attention (possibly non-trivial)  
    Potentially helpful Github [discussion](https://github.com/Dao-AILab/flash-attention/issues/1708#issuecomment-2987038903).
5. `python inference.py -i video.mp4`

If you have a long video, you can call the following script to split it into short clips to simulate streaming.  
`python preprocessing.py -i video.mp4`  
`python inference.py -i video_split/`

The program will print to screen and save to a file its analysis.  

---

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