import os
import cv2
import sys
import json
import math
import time
import logging
import platform
import numpy as np
from pathlib import Path
from skimage.draw import line

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

def loadJson(file):
    assert os.path.exists(file),f'file not exist: `{file}`'
    with open(file,'r',encoding='utf-8') as f:
        data = json.load(f)
    return data

class clockwise_angle_and_distance():
    '''
    A class to tell if point is clockwise from origin or not.
    This helps if one wants to use sorted() on a list of points.

    Parameters
    ----------
    point : ndarray or list, like [x, y]. The point "to where" we g0
    self.origin : ndarray or list, like [x, y]. The center around which we go
    refvec : ndarray or list, like [x, y]. The direction of reference

    use: 
        instantiate with an origin, then call the instance during sort
    reference: 
    https://stackoverflow.com/questions/41855695/sorting-list-of-two-dimensional-coordinates-by-clockwise-angle-using-python

    Returns
    -------
    angle
    
    distance
    

    '''
    def __init__(self, origin):
        self.origin = origin

    def __call__(self, point, refvec = [0, 1]):
        if self.origin is None:
            raise NameError("clockwise sorting needs an origin. Please set origin.")
        # Vector between point and the origin: v = p - o
        vector = [point[0]-self.origin[0], point[1]-self.origin[1]]
        # Length of vector: ||v||
        lenvector = np.linalg.norm(vector[0] - vector[1])
        # If length is zero there is no angle
        if lenvector == 0:
            return -math.pi, 0
        # Normalize vector: v/||v||
        normalized = [vector[0]/lenvector, vector[1]/lenvector]
        dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1] # x1*x2 + y1*y2
        diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1] # x1*y2 - y1*x2
        angle = math.atan2(diffprod, dotprod)
        # Negative angles represent counter-clockwise angles so we need to 
        # subtract them from 2*pi (360 degrees)
        if angle < 0:
            return 2*math.pi+angle, lenvector
        # I return first the angle because that's the primary sorting criterium
        # but if two vectors have the same angle then the shorter distance 
        # should come first.
        return angle, lenvector

class ContourParse():
    def __init__(self, contour, imgH=720, imgW=1280) -> None:
        self.contour = np.asarray(contour).reshape(-1,2)
        self.imgH = imgH
        self.imgW = imgW
        self.pts_num = len(self.contour)

    def split_LR(self):
        top_id = np.argmin(self.contour[...,1])
        bot_id = np.argmax(self.contour[...,1])
        max_id = np.max([top_id,bot_id])
        min_id = np.min([top_id,bot_id])
        cnt1_id = list(range(min_id,max_id+1))
        cnt2_id = list(range(max_id,self.pts_num))+list(range(0,min_id))
        cnt1 = self.contour[cnt1_id,...]
        cnt2 = self.contour[cnt2_id,...]
        if cnt1[-2][0]<cnt1[1][0]:
            left_cnt,right_cnt = cnt1,cnt2
        else:
            left_cnt,right_cnt = cnt2,cnt1
        return left_cnt,right_cnt

    def split_TB(self):
        pass

    def visual(self,contour=None,img=None):
        if contour is None:
            contour = self.contour
        mask = np.zeros([self.imgH,self.imgW,3]) if img is None else img
        cv2.fillPoly(mask,[contour],color=(0,255,0))
        try:
            cv2.imshow('visual',mask)
            cv2.waitKey(0)
        except:
            cv2.imwrite('visual.jpg',mask)
        return mask

def get_cross_pts(contour, pt1, pt2):
    cross_pts = []
    for pt in zip(*line(*pt1, *pt2)):
        if cv2.pointPolygonTest(contour, pt, False) == 0:  # 若点在轮廓上
            cross_pts.append(pt)
    cross_pts = np.array(cross_pts).reshape(-1,2)     
    if len(cross_pts)>2:
        max_id = np.argmax(cross_pts[...,0])
        min_id = np.argmin(cross_pts[...,0])
        cross_pts = cross_pts[[min_id,max_id],...]
    return cross_pts

def get_larger_contour(area_cnts:dict):
    assert sum(key in area_cnts.keys() for key in classes[:3]),f'some key not found'
    
    baffle_l = np.array(area_cnts[classes[0]])
    baffle_r = np.array(area_cnts[classes[1]])
    step = np.array(area_cnts[classes[2]])
    cnts = np.vstack((baffle_l,baffle_r,step))

    center_pt = np.array(cnts).mean(axis = 0).astype(np.int64) # get origin
    clock_ang_dist = clockwise_angle_and_distance(center_pt) # set origin
    cnts = sorted(cnts, key=clock_ang_dist) # use to sort
    cnts = np.asarray(cnts)

    img = visual_json(data=area_cnts)
    img = visual_contour(img,cnts)
    img=cv2.circle(img ,tuple(center_pt),2,(0,0,255),4) #画出重心
    cv2.imshow('img',img)
    cv2.waitKey(0)


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

def visual_json(img=None,data=None,file=None):
    assert data is not None or file is not None
    if data is None:
        data = loadJson(file)
    h,w = data['imgHeight'],data['imgWidth']
    if img is None:
        mask = np.zeros((h,w,3))
    else:
        mask = img.copy()
    for i,(key,value) in enumerate(data.items()):
        if key not in classes:
            continue
        cnt = np.array(data[key])
        visual_mask = cv2.fillPoly(mask,[cnt],color=colors[i])
    visual_mask = visual_mask.astype(np.uint8)
    # cv2.imshow('img',visual_mask)
    # cv2.waitKey(0)
    return visual_mask

def visual_contour(img,cnt,color=(127,127,127)):
    cv2.drawContours(img,[cnt],-1,color,3)
    return img


if __name__ == "__main__":
    # vid_path = r"data\4mm_53.mp4"
    # save_path = r"data/test.jpg"
    # get_img_from_video(vid_path,save_path=save_path,frame_id=650)
    get_larger_contour(loadJson('output/seg_result.json'))