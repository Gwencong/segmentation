# Segmentation with TensorRT and DeepStream
## 1. 主要内容
&emsp;&emsp;代码主要内容为对于分割模型的onnx推理和TRT导出与推理的示例  

* infer_TRT.py: 推理TRT模型
* export_TRT.py: 导出onnx模型为TRT模型
* infer_onnx.py 推理onnx模型
* export_onnx.py: 导出paddle模型为onnx模型(不是重点)  

&emsp;&emsp;分割的类别为左挡板、右挡板、梯路、背景四类，类别ID为0-3，模型推理输出为经过了softmax的概率值，比如输入为H×W×3的图片，输出为1×4×H×W的数组，其中的4表示对应的4个类别的概率值，因此后处理时需要对模型推理输出在第二个维度进行argmax得到H×W的预测mask，mask每个像素位置的值为该像素的预测类别的id,比如该像素被预测为背景类，而背景类id为3，那么该像素处的值为3，得到mask之后使用`get_contour_approx`和`result2json`函数进一步处理获取各个类别区域的轮廓，并将结果保存为json文件，以供3D入侵使用。

## 2. 使用说明

### (1) 导出onnx模型
&emsp;&emsp;由于模型是通过百度的[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)训练的，导出onnx模型按照PadddleSeg提供的[教程](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/docs/model_export_onnx_cn.md)导出onnx模型即可，当然，也可以通过本工程提供的export_onnx.py将paddle模型转换为onnx模型(需要[安装paddle框架和PaddleSeg库](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/docs/install_cn.md))，其用法如下:  
<details>
<summary>展开</summary>

```bash
python3 export_onnx.py \
    --config configs/pp_liteseg/pp_liteseg_stdc2_1024x512.yml \
    --model_path out/train/pp_litesegB/best_model/model.pdparams \
    --save_dir out/export/pp_liteseg \
    --width 1280 \
    --height 720 \
    --print_model
```
</details>  

### (2) 推理onnx模型
&emsp;&emsp;推理代码位于infer_onnx.py中，代码较为简单，运行时先设置好图片路径(img_path)和模型路径(onnx_path)，然后直接运行python文件即可。  

```bash
python3 infer_onnx.py
```

### (3) 导出TensorRT模型
&emsp;&emsp;导出TensorRT模型有两种方式，一种是使用TensorRT官方的trtexec转换工具，另外一种则是使用本工程提供的export_TRT.py脚本，根据需要设置好相关参数运行脚本即可，例如：  
```bash
python3 export_TRT.py --onnx weights/segformer.onnx
```

### (4) 推理TensorRT模型
&emsp;&emsp;工程中提供了使用TensorRT python API进行推理和使用DeepStream进行推理的代码示例。
#### a. 使用TensorRT python API进行推理：
&emsp;&emsp;更改infer_TRT.py文件中的图片路径和模型路径，然后直接运行即可。  
&emsp;&emsp;其中TRT_infer类的inference函数使用了装饰器以便测试速度，每次调用inference函数会默认执行10次推理并记录平均耗时，如果不需要测试速度，注释掉装饰器那一行即可。
#### b. 使用DeepStream进行推理
&emsp;&emsp;deepstream文件夹内的代码为使用DeepStream推理的代码，使用步骤如下：  
* 根据需要修改 deepstream/configs/segmentation_config_semantic.txt 文件中的内容，比如TensorRT模型路径(model-engine-file)
* 设置环境变量: 在命令行设置 
```bash
export DISPLAY=:0
```  
* 运行python文件
```bash
python3 deepstream/deepstream_segmentation.py
```  

## 3. 模型性能测试
&emsp;&emsp;根据PaddleSeg提供的模型性能测试结果，综合考虑选择了四种算法：Segformer-B、Deeplabv3+、FCN-HRNetW18、PP-Lite-Seg-B，并在Jetson Xavier NX上使用分别使用TensorRT python API和DeepStream测试了在输入为720×1280×3尺寸情况下，其速度和显存消耗，其结果如下表所示，综合二者拟采用FCN-HRNetW18，最终算法使用那种需要根据算法在实际分割数据上的表现而定。  


<p align="center"><font face="黑体" size=2.>表1 TensorRT Python API</font></p>

<div align="center">

|           | Segformer-B | Deeplabv3+ | FCN-HRNetW18 | PP-Lite-Seg-B |
|  :----:   | :---------: | :-------:  | :----------: | :-----------: |
|  速度(ms) |     583     |    410     |      217     |      148      |
| 显存占用(G)|      2.7   |     2.7     |      1.6     |     2.1      |

</div>

<p align="center"><font face="黑体" size=2.>表2 DeepStream</font></p>

<div align="center">

|            | Segformer-B | Deeplabv3+ | FCN-HRNetW18 | PP-Lite-Seg-B |
|  :----:    | :---------: | :-------:  | :----------: | :-----------: |
|  速度(FPS) |      2      |    3.4     |      9.3     |      10        |
| 显存占用(G)|      3      |     2.9     |      1.8     |      2.4      |

</div>

## 4. 模型下载
&emsp;&emsp;注意：供下载的模型仅用15张图片训练，精度很差，仅供参考，TensorRT的权重文件仅适用于Jetson Xavier NX上，TensorRT 8.0.1.6 版本

<p align="center"><font face="黑体" size=2.>表3 模型文件下载</font></p>

<div align="center">

|           | Segformer-B | Deeplabv3+ | FCN-HRNetW18 | PP-Lite-Seg-B |
|  :----:   | :---------: | :-------: | :----------: | :-----------: |
|  ONNX |     [ckpt](https://drive.google.com/file/d/1IzlahUU26lI-LaAAitLeYXPESh4z4uRO/view?usp=sharing)(107MB)     |    [ckpt](https://drive.google.com/file/d/11BnttuKZoxMgJnc0WZIpJQj5TYz1JnOB/view?usp=sharing)(105MB)    |      [ckpt](https://drive.google.com/file/d/17j_PJRIZHqjahCNgMKTwkmIX2_00WdmC/view?usp=sharing) &#124; [ckpt-dynamic](https://drive.google.com/file/d/15fII64YGDXhbBDEXJ9dcylluY5_2i5CF/view?usp=sharing)(38MB)      |      [ckpt](https://drive.google.com/file/d/1YYVCeMbt6sAXYlSFM996vfvBGczm26hK/view?usp=sharing)(48MB)      |
| TensorRT|      [ckpt](https://drive.google.com/file/d/1-puwvDEvU9_9IilhMBaSNTwpFoYh2Fvu/view?usp=sharing)(68MB)    |     [ckpt](https://drive.google.com/file/d/1wrU6ciUNA0euWrrf3b8I8Rxgj9PYmt08/view?usp=sharing)(56.6MB)     |     [ckpt](https://drive.google.com/file/d/19V7H_Ws3SZ6sDMBBfE8AEAWv-j2qr9te/view?usp=sharing)(31MB)      |      [ckpt](https://drive.google.com/file/d/1RGh59r8vAqWQQNcUMAJjLAKyAhSNJM4v/view?usp=sharing)(25MB)     |

</div>