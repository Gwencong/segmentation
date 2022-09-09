import cv2
import onnx
import onnxruntime
import numpy as np

from utils.utils import get_contour_approx,result2json

colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255,255,0], [0, 0, 0]]
classes = ['background','left baffle','right baffle','floor plate','step']
train_id_to_color = np.array(colors)
train_id_to_color = np.ascontiguousarray(train_id_to_color)


def normalize(im, mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)):
    # 图片归一化
    mean = np.array(mean)[np.newaxis, np.newaxis, :]    # [1,1,3]
    std = np.array(std)[np.newaxis, np.newaxis, :]      # [1,1,3]
    im = im.astype(np.float32, copy=False) / 255.0
    im -= mean
    im /= std
    return im

def preprocess(im):
    img = normalize(im)
    img = img.transpose((2, 0, 1))[::-1]    # HWC->CHW, BGR->RGB
    img = np.ascontiguousarray(img)
    img = np.expand_dims(img, 0)            # [C,H,W] -> [1,C,H,W]
    img = np.asarray(img,dtype=np.float32)
    return img

def postprocess(model_out):
    pred = np.argmax(model_out[0],axis=0)
    color_pred = train_id_to_color[pred].astype(np.uint8)
    return pred,color_pred

def infer_onnx(img_path,onnx_path):
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    # im = cv2.imread(img_path)
    im = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),-1)
    img = preprocess(im)
    
    ort_session = onnxruntime.InferenceSession(onnx_path,providers=['CPUExecutionProvider'])

    print("input shape:", img.shape)
    ort_inputs = {ort_session.get_inputs()[0].name: img}
    ort_output = ort_session.run(None, ort_inputs)[0]
    pred,color_pred = postprocess(ort_output)

    approxs = get_contour_approx(pred,im,visual=True)  # get contour points of roi area from segment result
    approxs.update(**{'imgHeight':im.shape[0],'imgWidth':im.shape[1]})
    result2json(approxs,'output/seg_result.json')       # save result to json file

    im = cv2.addWeighted(im,0.7,color_pred,0.3,0)

    cv2.imwrite('output/out_onnx_mask.jpg',color_pred)
    cv2.imwrite('output/out_onnx_fuse.jpg',im)

def infer_onnx_video(video_path,onnx_path):
    import time
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(onnx_path,providers=['CPUExecutionProvider'])
    
    cap = cv2.VideoCapture(video_path)

    frame_width  = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_count  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_fps    = cap.get(5)
    print('frame width: {}\nframe height: {}\nframe count {}\nFPS: {}'.format(frame_width,frame_height,frame_count,frame_fps))

    stillgo,frame = cap.read()
    count = 0
    while stillgo:
        start = time.time()
        print(count)
        img = preprocess(frame)
        ort_inputs = {ort_session.get_inputs()[0].name: img}
        ort_output = ort_session.run(None, ort_inputs)[0]
        pred,color_pred = postprocess(ort_output)

        approxs = get_contour_approx(pred,frame,visual=True)  # get contour points of roi area from segment result
        approxs.update(**{'imgHeight':frame.shape[0],'imgWidth':frame.shape[1]})
        result2json(approxs,'output/seg_result.json')       # save result to json file

        frame = cv2.addWeighted(frame,0.7,color_pred,0.3,0)

        cv2.imshow('img',cv2.resize(frame,(0,0),fx=0.7,fy=0.7))
        rest = max(1/frame_fps - (time.time()-start),1)
        k = cv2.waitKey(rest) & 0xff
        if k == 27:
            break
        
        stillgo,frame = cap.read()
        count += 1


if __name__ == "__main__":


    # img_path = "data/test.jpg"
    # img_path = "D:/my file/project/扶梯项目/code/OpticalFlow-DirectionJudge/data/test.jpg"
    # onnx_path = "weights/fcn_hrnetw18.onnx"
    # infer_onnx(img_path,onnx_path)

    # vid_path = r"data\down.mp4"
    vid_path = r"D:\my file\project\扶梯项目\自采集数据\50b6d76b9aa1ef95b5704a29d835f2b1.mp4"
    onnx_path = "weights/fcn_hrnetw18_dynamic.onnx"
    infer_onnx_video(vid_path,onnx_path)

    # img_path = r'D:\my file\project\扶梯项目\code\OpticalFlow-DirectionJudge\data\test.jpg'
    # onnx_path = r"D:\my file\project\扶梯项目\测试\图像分割\code\weights\fcn_hrnetw18.onnx"
    # infer_onnx(img_path,onnx_path)


 
