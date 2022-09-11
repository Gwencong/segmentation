import cv2
import sys
import json
import time
import logging
import platform
import numpy as np
from pathlib import Path

try:
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst
    
except Exception as e:
    print(repr(e))


SEG_COLORS = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 0, 0]]
colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255],[0, 0, 0]]
classes = ['left baffle','right baffle','step','floor plate','background']
train_id_to_color = np.array(colors)
train_id_to_color = np.ascontiguousarray(train_id_to_color)

class Timer(object):
    def __init__(self,iters=10) -> None:
        self.iters=iters

    def __call__(self, func):
        prefix = colorstr('TimeLogger')
        def wrapper(*args,**kwargs):
            print(f'{prefix} runing {func.__qualname__}() 10 times...')
            time_costs = []
            for i in range(self.iters):
                t1 = time.time()
                out = func(*args,**kwargs)
                cost = time.time() - t1
                time_costs.append(cost)
                print(f'[{i}] cost -> {cost*1000:.2f}ms')
            print(f'{prefix} Avarage time cost: {np.mean(time_costs)*1000:.2f}ms')
            return out
        return wrapper



def map_mask_as_display_bgr(mask):
    """ Assigning multiple colors as image output using the information
        contained in mask. (BGR is opencv standard.)
    """
    # getting a list of available classes
    m_list = list(set(mask.flatten()))

    shp = mask.shape
    bgr = np.zeros((shp[0], shp[1], 3),dtype=np.uint8)
    for idx in m_list:
        bgr[mask == idx] = SEG_COLORS[idx]
    return bgr

## gst

def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        sys.stdout.write("End-of-stream\n")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        sys.stderr.write("Warning: %s: %s\n" % (err, debug))
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write("Error: %s: %s\n" % (err, debug))
        loop.quit()
    return True

def is_aarch64():
    return platform.uname()[4] == "aarch64"

def make_element(factoryname, name, detail=""):
    """ Creates an element with Gst Element Factory make.
        Return the element  if successfully created, otherwise print
        to stderr and return None.
    """
    print("Creating {}({}) \n".format(name, factoryname))
    elm = Gst.ElementFactory.make(factoryname, name)
    if not elm:
        sys.stderr.write("Unable to create {}({}) \n".format(name, factoryname))
        if detail:
            sys.stderr.write(detail)
    return elm

def cb_newpad(decodebin, decoder_src_pad, data):
    print("In cb_newpad\n")
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    print("gstname=", gstname)
    if gstname.find("video") != -1:
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        # print("features=", features)
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")

def decodebin_child_added(child_proxy, Object, name, user_data):
    print("Decodebin child added:", name, "\n")
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)
    if is_aarch64() and name.find("nvv4l2decoder") != -1:
        print("Seting bufapi_version\n")
        Object.set_property("bufapi-version", True)

def create_source_bin(index, uri):
    print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name = "source-bin-%02d" % index
    print(bin_name)
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri", uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin

def set_logging(rank=-1, verbose=True):
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if (verbose and rank in [-1, 0]) else logging.WARN)

def file_size(file):
    # Return file size in MB
    return Path(file).stat().st_size / 1e6

def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

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
        # cv2.imshow('test',mask)
        # cv2.waitKey(0)
        contours,hierarchy = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            
            # if contours number greater than 1, filter according to contour area
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



def get_img_from_video(vid_path,frame_id=0,save_path=None):
    cap = cv2.VideoCapture(vid_path)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(5)
    print('frame width: {}\nframe height: {}\nframe count {}\nFPS: {}'.format(frame_width,frame_height,frame_count,fps))
    count = 0

    if cap.isOpened():
        ret,frame = cap.read()
    else:
        ret = False 
    while ret:
        if count == frame_id:
            break
        ret,frame  = cap.read()
        if frame is not None: 
            pass
        count+=1
    cap.release()
    if save_path and frame is not None:
        cv2.imwrite(save_path,frame)
        print('frame {} has been save to `{}`'.format(frame_id,save_path))

    return frame

if __name__ == "__main__":
    vid_path = r"D:\my file\project\扶梯项目\鲁邦通数据\8mm_20220427163046407.mp4"
    save_path = r"data/test.jpg"
    get_img_from_video(vid_path,save_path=save_path,frame_id=14)