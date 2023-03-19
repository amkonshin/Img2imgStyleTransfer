# Img2imgStyleTransfer
Img2imgStyleTransfer
### Requirements
torch==1.8.1
torchvision==0.9.1

### Testing 
Для запуска проекта необходимо скачать веса моделей в каталог ./experiments/: [vgg-model](https://drive.google.com/file/d/1BinnwM5AmIcVubr16tPTqxMjUCE8iu5M/view?usp=sharing),   [Transformer_module](https://drive.google.com/file/d/1dnobsaLeE889T_LncCkAA2RkqzwsfHYy/view?usp=sharing)   <br> 

```
python3 test.py --content input/content/giga.jpg --style input/style/style1.jpg
```
Результаты обработки хранятся в ./output


### References 

Paper Link [pdf](https://arxiv.org/abs/2105.14576)<br> 
```
@inproceedings{deng2021stytr2,
      title={StyTr^2: Image Style Transfer with Transformers}, 
      author={Yingying Deng and Fan Tang and Weiming Dong and Chongyang Ma and Xingjia Pan and Lei Wang and Changsheng Xu},
      booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2022},
}