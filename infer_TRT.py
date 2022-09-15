import json
import cv2
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

from pathlib import Path
from utils.utils import Timer,colorstr

colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255,255,0], [0, 0, 0]]
classes = ['left baffle','right baffle','step','floor plate','background']
train_id_to_color = np.array(colors)
train_id_to_color = np.ascontiguousarray(train_id_to_color)


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TRT_Infer():
    def __init__(self,engine_path,shape=[1,3,640,640],num_classes=5) -> None:
        self.shape = shape              # 模型的输入图片形状
        self.num_classes = num_classes  # 类别数量
        self.logger = trt.Logger(trt.Logger.WARNING)    # tensorrt日志记录器
        self.runtime = trt.Runtime(self.logger)         # tensorrt运行时
        self.engine = self.load_engine(engine_path,self.runtime)    # 导入TRT模型
        self.context = self.engine.create_execution_context()       # 获取执行上下文
        self.stream = cuda.Stream()                                 # 获取数据处理流
        self.inputs, self.outputs, self.bindings = self.allocate_buffers_dynamic(self.context,self.shape)
    
    def warmup(self,iters=5):
        prefix = colorstr('TRT_Infer')
        print(f'{prefix} warmup {iters} times...')
        for i in range(iters):
            b,c,h,w = self.shape
            dummy = np.random.randint(0,255,(h,w,c),dtype=np.uint8)
            img = self.preprocess(dummy, self.inputs[0].host)
            [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
            self.stream.synchronize()
            model_out = self.outputs[0].host
            self.postprocess(model_out, img.shape)

    def normalize(self, im, mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)):
        # 图片归一化
        mean = np.array(mean)[np.newaxis, np.newaxis, :]    # [1,1,3]
        std = np.array(std)[np.newaxis, np.newaxis, :]      # [1,1,3]
        im = im.astype(np.float32, copy=False) / 255.0
        im -= mean
        im /= std
        return im

    def load_engine(self,engine_path,runtime):
        # 加载TRT模型
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        return engine

    def allocate_buffers_dynamic(self,context,shape):
        # 分配device内存
        inputs = []
        outputs = []
        bindings = []
        context.set_binding_shape(0,shape)   # Dynamic Shape 模式需要绑定真实数据形状
        engine = context.engine
        for binding in engine:
            ind = engine.get_binding_index(binding)
            size = trt.volume(context.get_binding_shape(ind)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings

    def preprocess(self, im, pagelocked_buffer):
        # 预处理, 并将预处理结果拷贝到分配的主机内存上
        img = self.normalize(im)
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = np.expand_dims(img, 0)
        img = np.asarray(img,dtype=np.float32)
        np.copyto(pagelocked_buffer, (img.astype(trt.nptype(trt.float32)).ravel()))
        return img

    @Timer(10)
    def inference(self,im,logits=False):
        # preprocess
        img = self.preprocess(im, self.inputs[0].host)
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
        # Run inference.
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
        # Synchronize the stream
        self.stream.synchronize()
        # Return only the host outputs.
        model_out = self.outputs[0].host
        # postprocess
        mask,color_mask = self.postprocess(model_out, img.shape, logits)
        return mask,color_mask
    
    def postprocess(self,model_out,shape,logits=False):
        # 后处理, argmax获取每个像素的预测类别
        b,c,h,w = shape
        pred = model_out.reshape(b,self.num_classes,h,w)[0]         # [4,H,W]
        if not logits:
            pred = np.argmax(pred,axis=0).astype(np.int64)          # [H,W]
            color_pred = train_id_to_color[pred].astype(np.uint8)   # [H,W,3]
        else:
            color_pred = None
        return pred, color_pred


def get_contour_approx(pred,img,visual=False):
    '''根据预测的mask获取扶梯左右挡板、梯路的轮廓\n
    Args:
        pred: 预测的mask, 尺寸: [H,W], 每个像素的值为0-3, 对于类别id
        img: 原图, 可视化用
    Return: 
        approxs: 获取到的轮廓点集, list, 有三个元素, 对应左右挡板和梯路的区域轮廓
    '''
    h,w = pred.shape[:2]
    approxs = {i:[] for i in classes[:4]}
    for i in range(4):
        mask = np.where(pred==i,255,0).astype(np.uint8)
        contours,hierarchy = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            areas = [cv2.contourArea(contour) for contour in contours]
            areas_ids = np.array([(j,area) for j,area in enumerate(areas) if 0.01<area/(h*w)<0.8]) # filter
            if len(areas_ids) == 0:
                print('Warning: contour areas is out of range, the segmentation result may be incorrect')
                indexes = np.arange(len(areas))
            else:
                areas = areas_ids[:,1]
                indexes = areas_ids[:,0]
            idx = int(indexes[np.argmax(areas)]) 
            contour = contours[idx] # select contour with max area 
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour,epsilon,True) # smoothing
            # approx = cv2.convexHull(contour)
            approxs[classes[i]] = approx.reshape(-1,2).tolist() if isinstance(approx,np.ndarray) else approx
            if visual:
                cv2.drawContours(img,[approx],-1,(0,255,255),thickness=4)
        else:
            print(f'no contour is found for class `{classes[i]}`')
    if visual:
        cv2.imwrite('output/out-trt-aprroxs.jpg',img) 
    return approxs

def result2json(data,file='seg.json'):
    assert file.endswith('.json'),f'invalid file name `{file}`'
    with open(file,'w',encoding='utf-8') as f:
        json.dump(data,f,indent=2,ensure_ascii=False)
    print(f'segment result has been saved to `{file}`')


def infer_trt(img_path,model_path):
    '''单张图片推理
    
    Arguments:
        img_path: 输入图片文件路径
        model_path: TensorRT模型文件路径
    Returns:
        None
    '''
    img = cv2.imread(img_path)
    input_shape = [1,3,img.shape[0],img.shape[1]]
    model = TRT_Infer(engine_path=model_path,shape=input_shape)
    model.warmup(5)
    mask,color_mask = model.inference(img)

    vis_img = img.copy()
    approxs = get_contour_approx(mask,vis_img,visual=True)  # get contour points of roi area from segment result
    approxs.update(**{'imgHeight':img.shape[0],'imgWidth':img.shape[1]})
    result2json(approxs,'output/seg_result.json')       # save result to json file
    
    img = cv2.addWeighted(vis_img,0.7,color_mask,0.3,0)
    cv2.imwrite('output/out_trt_mask.jpg',color_mask)
    cv2.imwrite('output/out_trt_fuse.jpg',img)

def infer_trt_multi(imgs,model_path,mode='mean'):
    '''多张图片推理

    Arguments:
        imgs: 输入图片的list, 每个元素是一个图片数组, 尺寸必须相同
        model_path: TensorRT模型文件路径
        mode: 多帧图片推理结果的处理方式, mean: 取平均, max: 取最大
    Returns:
        None
    '''
    assert mode in ['mean','max'],"Argument `mode` invalid, should one of ['mean','max']"

    # model init
    img0 = imgs[0]
    input_shape = [1,3,img0.shape[0],img0.shape[1]]
    model = TRT_Infer(engine_path=model_path,shape=input_shape)
    model.warmup(5)

    # inference
    masks = []
    for img in imgs:
        mask,_ = model.inference(img,logits=True)           # [4,H,W]
        masks.append(mask)
    mask = np.mean(masks,axis=0) if mode == 'mean' else np.max(masks,axis=0)    # [H,W]
    mask = np.argmax(mask,axis=0).astype(np.int64)          # [H,W]
    color_mask = train_id_to_color[mask].astype(np.uint8)   # [H,W,3]

    vis_img = img0.copy()
    approxs = get_contour_approx(mask,vis_img,visual=True)  # get contour points of roi area from segment result
    approxs.update(**{'imgHeight':img0.shape[0],'imgWidth':img0.shape[1]})
    result2json(approxs,'output/seg_result.json')       # save result to json file
    
    img = cv2.addWeighted(vis_img,0.7,color_mask,0.3,0)
    cv2.imwrite('output/out_trt_mask.jpg',color_mask)
    cv2.imwrite('output/out_trt_fuse.jpg',img)



if __name__ == "__main__":
    # # 测试范例1: 单张图片推理
    # img_path = 'data/test.jpg'
    # model_path = 'weights/fcn_hrnetw18.trt'
    # infer_trt(img_path,model_path)

    # 测试范例2: 多张图片推理
    img_path = 'data/test.jpg'
    img = cv2.imread(img_path)
    imgs = [img for _ in range(5)]
    model_path = 'weights/fcn_hrnetw18.trt'
    infer_trt_multi(imgs,model_path,mode='mean')