新建文件夹train、saved和gen_images三个文件夹
将raw文件和mhd文件放入train文件夹中，saved文件夹保存由mhd文件生成的png图片，gen_images文件夹保存训练出来的图片。
首先运行main.py，生成png图片。
由于此时没有保存模型，因此要首先在config.py中将LOAD_MODEL更改为False。
上述完成后，即可运行train.py，生成图片和模型。
预计训练20epochs可以初见成效。