# PeekabooSAM installtiaon

```
conda create -n sam python=3.10
cd sam2
pip install -e .
pip install -e ".[notebooks]"
pip install -r requirements.txt
```

# Get checkpiitns
```
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

# Demo on custom video

To run the demo with your custom video, 

```
CUDA_VISIBLE_DEVICES=1 python track_foreground_object.py --video-path ../data/examples/videos/person_2.mp4 --output-path ../outputs/p_2.mp4

```

SAMURAI is built on top of SAM 2 by Meta FAIR.

https://yangchris11.github.io/samurai/