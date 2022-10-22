import os
import cv2
import time
import random
import matplotlib
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
from tqdm import tqdm
from pathlib import Path

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        self.palette = [self.hex2rgb(c) for c in matplotlib.colors.TABLEAU_COLORS.values()]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

colors = Colors()  # create instance for 'from utils.plots import colors'


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class Pose_TRT():
    def __init__(self, engine_path, device=None, input_shape=None,conf=0.4,iou=0.5,engine=None) -> None:
        assert os.path.exists(engine_path),f'engine file `{engine_path}` is not exist!'
        self.iou = iou
        self.conf = conf
        self.device = device
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(engine_path,self.runtime) if engine is None else engine
        self.context = self.engine.create_execution_context()
        self.input_shape,self.dynamic = self.get_shape(self.engine,input_shape)
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.context,self.input_shape)
    

    def call_old(self, img, visual=False, *args, **kwds):
        b,c,h,w = self.input_shape
        np.copyto(self.inputs[0].host, (img.astype(trt.nptype(trt.float32)).ravel()))
        out = self.do_inference_v2()
        if len(out)>1:
            nmsed_indices,nmsed_boxes,nmsed_poses,nmsed_scores = out
            nmsed_indices = nmsed_indices.reshape(b,-1,3)
            nmsed_boxes = nmsed_boxes.reshape(b,-1,4)
            nmsed_poses = nmsed_poses.reshape(b,-1,51)
            nmsed_scores = nmsed_scores.reshape(b,-1,1)
            nmsed_classes = np.zeros_like(nmsed_scores)
            out = np.concatenate([nmsed_boxes,nmsed_scores,nmsed_classes,nmsed_poses],axis=-1)
        else:
            out = out[0]
        out = out.reshape(b,-1,57)
        return out


    def __call__(self, im, visual=False, *args, **kwds):
        img  = self.preprocess(im,self.inputs[0].host)
        out  = self.do_inference_v2()
        pred = self.postprocess(out,im,img,has_nms=False,draw=visual)
        boxes, confs, kpts = pred
        return boxes, confs, kpts


    def load_engine(self,engine_path,runtime):
        trt.init_libnvinfer_plugins(None,'')    # load all avalilable official pluging
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        return engine

    def warmup(self,times=5):
        # print(f'warmup {times} times...')
        b,c,h,w = self.input_shape
        dummyinput = np.random.randint(0,255,(h,w,c),np.uint8)
        self.preprocess(dummyinput,self.inputs[0].host)
        for i in range(times):
            self.do_inference_v2()

    def get_shape(self,engine,input_shape=None):
        dynamic = False
        shape = engine.get_binding_shape(0)
        if -1 in shape:
            dynamic = True
            shapes = engine.get_profile_shape(profile_index=0, binding=0)
            min_shape,opt_shape,max_shape = shapes
            shape = input_shape if input_shape is not None else opt_shape
            print(f'engine shape range:\n   MIN: {min_shape}\n  OPT: {opt_shape}\n  MAX: {max_shape}')
            print(f'set engine shape to: {input_shape}')
        else:
            assert shape == input_shape, f'engine shape `{shape}` is not compatible with given shape `{input_shape}`'
        return shape,dynamic

    def allocate_buffers(self,context,shape):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        if self.dynamic:
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
        return inputs, outputs, bindings, stream

    def preprocess(self, im, pagelocked_buffer):
        img = letterbox(im, self.input_shape[2:], stride=64, auto=False)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = np.expand_dims(img, 0)
        img = img/255
        img = np.asarray(img,dtype=np.float32)
        np.copyto(pagelocked_buffer, (img.astype(trt.nptype(trt.float32)).ravel()))
        return img

    def do_inference_v2(self):
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
        # Run inference.
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
        # Synchronize the stream
        self.stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in self.outputs]
    
    def postprocess(self,model_out,im0,img,has_nms=False,draw=True):
        b,c,h,w = self.input_shape
        if not has_nms:
            pred = model_out[-1].reshape(b,-1,57)
            pred = self.non_max_suppression(pred, self.conf, self.iou)
        else:
            nmsed_indices,nmsed_boxes,nmsed_poses,nmsed_scores = model_out
            nmsed_indices = nmsed_indices.reshape(b,-1,3)
            nmsed_boxes = nmsed_boxes.reshape(b,-1,4)
            nmsed_boxes[0] = self.xywh2xyxy(nmsed_boxes[0])
            nmsed_poses = nmsed_poses.reshape(b,-1,51)
            nmsed_scores = nmsed_scores.reshape(b,-1,1)
            tmp = np.zeros_like(nmsed_scores)
            pred = np.concatenate([nmsed_boxes,nmsed_scores,tmp,nmsed_poses],axis=-1)
            keep = np.unique(nmsed_indices[...,2]).size
            pred = pred[:,:keep,:]
            # print(pred.shape)
        boxes,confs,kepts = [], [], []
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                self.scale_coords((h,w), det[:, :4], im0.shape, kpt_label=False)
                self.scale_coords((h,w), det[:, 6:], im0.shape, kpt_label=True, step=3)

                boxes.append(det[:,:4])
                confs.append(det[:,4:5])
                kepts.append(det[:,6:])
                # Write results
                if not draw: continue
                for det_index, (*xyxy, conf, cls) in enumerate(reversed(det[:,:6])):
                    c = int(cls)  # integer class
                    kpts = det[det_index, 6:]
                    plot_one_box(xyxy, im0, color=colors(c, True), line_thickness=3, kpt_label=True, kpts=kpts, steps=3, orig_shape=im0.shape[:2])
        return boxes, confs, kepts

    def non_max_suppression(self, predictions, conf_thres=0.5, nms_thres=0.4):
        """
        description: Removes detections with lower object confidence score than 'conf_thres' and performs
        Non-Maximum Suppression to further filter detections.
        param:
            prediction: detections, (x1, y1, x2, y2, conf, cls_id)
            origin_h: original image height
            origin_w: original image width
            conf_thres: a confidence threshold to filter detections
            nms_thres: a iou threshold to filter detections
        return:
            boxes: output after nms with the shape (x1, y1, x2, y2, conf, cls_id)
        """
        output = [np.zeros((0, 57))] * predictions.shape[0]
        for i,prediction in enumerate(predictions):
            # Get the boxes that score > CONF_THRESH
            boxes = prediction[prediction[:, 4] >= conf_thres]
            # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
            # boxes[:, :4] = self.xywh2xyxy(origin_h, origin_w, boxes[:, :4])
            boxes[:, :4] = self.xywh2xyxy(boxes[:, :4])
            # clip the coordinates
            # boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w -1)
            # boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w -1)
            # boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h -1)
            # boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h -1)
            # Object confidence
            confs = boxes[:, 4] * boxes[:, 5]
            # Sort by the confs
            boxes = boxes[np.argsort(-confs)]
            # Perform non-maximum suppression
            keep_boxes = []
            while boxes.shape[0]:
                large_overlap = self.bbox_iou(np.expand_dims(boxes[0, :4], 0), boxes[:, :4]) > nms_thres
                # label_match = boxes[0, -1] == boxes[:, -1]
                # Indices of boxes with lower confidence scores, large IOUs and matching labels
                # invalid = large_overlap & label_match
                invalid = large_overlap
                keep_boxes += [boxes[0]]
                boxes = boxes[~invalid]
            boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
            output[i] = boxes
        return output

    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        """
        description: compute the IoU of two bounding boxes
        param:
            box1: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
            box2: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))            
            x1y1x2y2: select the coordinate format
        return:
            iou: computed iou
        """
        box1, box2 = box1.astype(np.float32), box2.astype(np.float32)
        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        # Get the coordinates of the intersection rectangle
        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        # Intersection area
        inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * \
                     np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
        
        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y

        return y

    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None, kpt_label=False, step=2):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0]
            pad = ratio_pad[1]
        if isinstance(gain, (list, tuple)):
            gain = gain[0]
        if not kpt_label:
            coords[:, [0, 2]] -= pad[0]  # x padding
            coords[:, [1, 3]] -= pad[1]  # y padding
            coords[:, [0, 2]] /= gain
            coords[:, [1, 3]] /= gain
            self.clip_coords(coords[0:4], img0_shape)
            #coords[:, 0:4] = coords[:, 0:4].round()
        else:
            coords[:, 0::step] -= pad[0]  # x padding
            coords[:, 1::step] -= pad[1]  # y padding
            coords[:, 0::step] /= gain
            coords[:, 1::step] /= gain
            self.clip_coords(coords, img0_shape, step=step)
            #coords = coords.round()
        return coords

    def clip_coords(self,boxes, img_shape, step=2):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        boxes[:, 0::step] = np.clip(boxes[:, 0::step], 0, img_shape[1])
        boxes[:, 1::step] = np.clip(boxes[:, 1::step], 0, img_shape[0])


class PersonCounter:
    def __init__(self,weight,imgNum,vidPath,saveDir,debug=False) -> None:
        self.weight = weight
        self.imgNum = imgNum
        self.vidPath = vidPath
        self.saveDir = saveDir
        self.debug = debug
        self.PersonDet = Pose_TRT(weight,input_shape=None,conf=0.4,iou=0.5)
        self.parse_video(vidPath)
        Path(saveDir).mkdir(parents=True,exist_ok=True)

        if debug:
            vid_fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
            out_path = 'output/debug/out.mp4'
            Path(out_path).parent.mkdir(parents=True,exist_ok=True)
            self.vid_writer = cv2.VideoWriter(out_path,vid_fourcc,self.vid_fps,(self.vid_w,self.vid_h))

    def __del__(self):
        if hasattr(self,'vid_writer'):
            self.vid_writer.release()

    def parse_video(self,path):
        assert os.path.exists(path),f'video `{path}` is not exist!'
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.vid_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.vid_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vid_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vid_name = os.path.splitext(os.path.basename(path))[0]

        print(f'Video from {path}')
        print(f'Height: {self.vid_h}\nWidth:{self.vid_w}\nFPS:{self.vid_fps}')
    
    def getNoPerson(self):
        start = time.time()
        noPerson,noPersonImg = 0,[]
        stillgo,im = self.cap.read()
        pbar = tqdm(desc='FPS:0',total=self.nframes)
        while stillgo:
            begin = time.time()
            self.frame += 1
            img = im.copy()
            
            boxes, confs, kpts = self.PersonDet(img,self.debug)
            if (len(boxes) == 0) and (noPerson < self.imgNum):
                img_name = f'{self.vid_name}_{self.frame}.jpg'
                img_path = os.path.join(self.saveDir,img_name)
                cv2.imwrite(img_path, im)
                noPerson += 1
                noPersonImg.append(im)
            if self.debug:
                self.vid_writer.write(img)
            stillgo,im = self.cap.read()
            use_time = time.time() - start
            avgFPS = int(1/(use_time/self.frame))
            FPS = int(1/(time.time() - begin))
            pbar.set_description(F'[{noPerson}/{self.imgNum}] avgFPS:{avgFPS}, FPS:{FPS}, PersonNum:{len(boxes)}')
            pbar.update()

            if (noPerson >= self.imgNum) and not self.debug:
                break
        pbar.close()
        if noPerson < self.imgNum:
            print(F'try to extract {self.imgNum} no person images, but get {noPerson}')
        return noPersonImg



def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def plot_one_box(x, im, color=None, label=None, line_thickness=3, kpt_label=False, kpts=None, steps=2, orig_shape=None):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, (255,0,0), thickness=tl*1//3, lineType=cv2.LINE_AA)
    if label:
        if len(label.split(' ')) > 1:
            label = label.split(' ')[-1]
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 6, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 6, [225, 255, 255], thickness=tf//2, lineType=cv2.LINE_AA)
    if kpt_label:
        plot_skeleton_kpts(im, kpts, steps, orig_shape=orig_shape)

def plot_skeleton_kpts(im, kpts, steps, orig_shape=None):
    #Plot the skeleton and keypointsfor coco datatset
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    radius = 5
    num_kpts = len(kpts) // steps

    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
            if steps == 3:
                conf = kpts[steps * kid + 2]
                if conf < 0.5:
                    continue
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)

    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0]-1)*steps]), int(kpts[(sk[0]-1)*steps+1]))
        pos2 = (int(kpts[(sk[1]-1)*steps]), int(kpts[(sk[1]-1)*steps+1]))
        if steps == 3:
            conf1 = kpts[(sk[0]-1)*steps+2]
            conf2 = kpts[(sk[1]-1)*steps+2]
            if conf1<0.5 or conf2<0.5:
                continue
        if pos1[0]%640 == 0 or pos1[1]%640==0 or pos1[0]<0 or pos1[1]<0:
            continue
        if pos2[0] % 640 == 0 or pos2[1] % 640 == 0 or pos2[0]<0 or pos2[1]<0:
            continue
        cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)



def sample01():
    engine_path = '/home/gwc/gwc/code/yolo-pose-tensorrt/weights/yolov5s6_pose_ti_lite_FP16.engine'
    input_shape = [1, 3, 640, 640]
    confidence  = 0.4
    iou_thresh  = 0.5
    pose_estimater = Pose_TRT(engine_path,input_shape=input_shape,conf=confidence,iou=iou_thresh)

    img_path = 'test01.jpg'
    im  = cv2.imread(img_path)
    img  = pose_estimater.preprocess(im,pose_estimater.inputs[0].host)
    out  = pose_estimater.do_inference_v2()
    pred = pose_estimater.postprocess(out,im,img,has_nms=False,draw=True)
    boxes, confs, kpts = pred
    print(boxes)
    print(confs)
    print(kpts)
    cv2.imwrite('pose.jpg',im)

def sample02():
    from tqdm import tqdm
    import time
    start = time.time()
    engine_path = '/home/gwc/code/yolo-pose-escalator/weights/yolov5l6_pose_mix-FP16.trt'
    input_shape = [1, 3, 1280, 1280]
    confidence  = 0.4
    iou_thresh  = 0.5
    pose_estimater = Pose_TRT(engine_path,input_shape=input_shape,conf=confidence,iou=iou_thresh)

    vid_path = 'data/4mm_04.mp4'
    vid_cap = cv2.VideoCapture(vid_path)
    fps,h,w = vid_cap.get(cv2.CAP_PROP_FPS),vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT),vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps,h,w = int(fps),int(h),int(w)
    print(f'Video from {vid_path}')
    print(f'Height: {h}\nWidth:{w}\nFPS:{fps}')
    # vid_writer = cv2.VideoWriter('data/out.avi',cv2.VideoWriter_fourcc("X", "V", "I", "D"),fps,(w,h))
    vid_writer = cv2.VideoWriter('data/out.mp4',cv2.VideoWriter_fourcc("m", "p", "4", "v"),fps,(w,h))
    stillgo,im = vid_cap.read()
    count = 0
    pbar = tqdm(desc='FPS:0',total=int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    while stillgo:
        begin = time.time()
        count += 1

        img  = pose_estimater.preprocess(im,pose_estimater.inputs[0].host)
        out  = pose_estimater.do_inference_v2()
        pred = pose_estimater.postprocess(out,im,img,has_nms=False,draw=True)
        boxes, confs, kpts = pred
        vid_writer.write(im)
        stillgo,im = vid_cap.read()
        

        use_time = time.time() - start
        avgFPS = int(1/(use_time/count))
        FPS = int(1/(time.time() - begin))
        pbar.set_description(F'avgFPS:{avgFPS}, FPS:{FPS}')
        pbar.update()
        
    pbar.close()
    vid_writer.release()

def sample03():
    weight = 'weights/yolov5l6_pose_mix-dynamic-FP16.trt'
    imgNum = 5
    vidPath = 'data/4mm_12.mp4'
    saveDir = 'data/imgs'
    counter = PersonCounter(weight,imgNum,vidPath,saveDir,debug=True)
    noPersonImgs = counter.getNoPerson()

if __name__ == "__main__":
    sample03()
    pass