# TEST
Download the pretrained weights from [here](https://drive.google.com/file/d/1bCIKOGNlKcGgObg5AeErUHkRTMuv0HHZ/view?usp=sharing) and put it into `pretrained/opendelight`.
The provided pretrained weights are identical to the `OpenDelight` network in the paper, i.e., trained on the mix of FaceOLAT and our private Light Stage scan dataset.

Run the following command to test on the example dataset in `misc/test_ffhq`:

```
python test.py \
    --data_root misc/test_ffhq \
    --save_root workspace/test_ffhq_results \
    --device 0
```

Then, you can check the delighting results in `workspace/test_ffhq_results`.
