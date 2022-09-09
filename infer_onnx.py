import cv2
import onnx
import onnxruntime
import numpy as np

colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255],[255,255,0], [0, 0, 0]]
classes = ['left baffle','right baffle','step','floor plate','background',]
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
    im = cv2.imread(img_path)
    img = preprocess(im)
    
    ort_session = onnxruntime.InferenceSession(onnx_path,providers=['CPUExecutionProvider'])

    print("input shape:", img.shape)
    ort_inputs = {ort_session.get_inputs()[0].name: img}
    ort_output = ort_session.run(None, ort_inputs)[0]
    pred,color_pred = postprocess(ort_output)

    im = cv2.addWeighted(im,0.7,color_pred,0.3,0)

    cv2.imwrite('output/out_onnx_mask.jpg',color_pred)
    cv2.imwrite('output/out_onnx_fuse.jpg',im)
    

if __name__ == "__main__":
    img_path = "data/test.jpg"
    onnx_path = "weights/fcn_hrnetw18.onnx"
    infer_onnx(img_path,onnx_path)

 
