## Install

to run locally do the following:
```bash
git clone https://github.com/justinpinkney/stable-diffusion.git
cd stable-diffusion
git checkout 1c8a598f312e54f614d1b9675db0e66382f7e23c
python -m venv .venv --prompt sd
. .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
git clone https://github.com/amkonshin/Img2imgStyleTransfer.git
```
to run style transfer:
```bash
python  Img2imgStyleTransfer/local_image_mixer.py --content scripts/style/img1.jpeg --style scripts/style/img2.jpeg --steps 35
```
