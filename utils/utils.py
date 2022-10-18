import os
import cv2
import sys
import json
import time
import logging
import platform
import numpy as np
from pathlib import Path
# from skimage.draw import line

try:
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst
    
except Exception as e:
    print(repr(e))


SEG_COLORS = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255,255,0], [0, 0, 0]]
classes = ['left baffle','right baffle','step','floor plate','background']
train_id_to_color = np.array(SEG_COLORS)
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


class ContourParse():
    class Line:
        def __init__(self,pt1,pt2,id=0) -> None:
            self.pt1 = pt1
            self.pt2 = pt2
            self.id = id
            self.x_low = min(pt1[0],pt2[0])
            self.x_up  = max(pt1[0],pt2[0])
            self.y_low = min(pt1[1],pt2[1])
            self.y_up  = max(pt1[1],pt2[1])
            self.k, self.b = self._get_kb(pt1,pt2)
        
        def _get_kb(self,pt1,pt2):
            # 获取线段的斜率和截距
            x1,y1 = pt1
            x2,y2 = pt2
            k = None if x1 == x2 else (y1-y2)/(x1-x2)
            b = None if k is None else y1-k*x1
            return k,b

        def is_inrange(self,x=None,y=None):
            x_inrange = True
            y_inrange = True
            if x is not None:
                x_inrange = self.x_low <= x <= self.x_up
            if y is not None:
                y_inrange = self.y_low <= y <= self.y_up
            return x_inrange and y_inrange
  
        def __call__(self, x=None,y=None, *args, **kwds):
            '''输入x坐标, 则计算对应的y坐标; 输入y坐标, 则计算对应的x坐标\n
            '''
            assert (x is None) ^ (y is None ),'x and y should not be specified or None at the same time'
            result = 0
            if x is not None:  
                if self.k is None:
                    result = None       # 此时线段与y轴平行，给定x坐标无法求出y坐标
                else:
                    result = self.k*x + self.b
            if y is not None:
                if self.k is None:
                    result = self.x_low # 此时线段与y轴平行，x坐标为固定值
                elif self.k == 0:
                    result = None       # 此时线段与y轴平行，给定y坐标无法求出x坐标
                else:
                    result = (y-self.b)/self.k
            return result

        def judgeLR(self,x,y):
            if self.k is None:
                res = True if x < self.x_low else False
            else:
                out = y - self.k*x - self.b
                if self.k > 0:
                    res = True if out>0 else False
                else:
                    res = False if out>0 else True
            return res


    def __init__(self, contour, imgH=720, imgW=1280, clock_wise=True, cnt_type='step') -> None:
        self.contour = np.asarray(contour).reshape(-1,2)
        self.imgH = imgH
        self.imgW = imgW
        self.clock_wise = clock_wise
        self.pts_num = len(self.contour)
        self.cnt_type = cnt_type
        self.left_cnt,self.left_ids, self.right_cnt,self.right_ids,\
            self.top_cnt,self.top_ids, self.bot_cnt,self.bot_ids = self.split()

    def split(self):
        # 1.分割上下轮廓
        top_id = np.argmin(self.contour[...,1])
        bot_id = np.argmax(self.contour[...,1])
        top_cnt,top_ids = self.search_pt(self.contour,top_id)
        bot_cnt,bot_ids = self.search_pt(self.contour,bot_id)
        if self.clock_wise:
            if top_cnt[0][0] > top_cnt[-1][0]:
                top_cnt = list(reversed(top_cnt))
            if bot_cnt[0][0] < bot_cnt[-1][0]:
                bot_cnt = list(reversed(bot_cnt))
        else:
            if top_cnt[0][0] < top_cnt[-1][0]:
                top_cnt = list(reversed(top_cnt))
            if bot_cnt[0][0] > bot_cnt[-1][0]:
                bot_cnt = list(reversed(bot_cnt))
        top_cnt = np.array(top_cnt)
        bot_cnt = np.array(bot_cnt)

        # 寻找轮廓中间线
        top_center = np.mean(top_cnt[[0,-1],...],axis=0)
        bot_center = np.mean(bot_cnt[[0,-1],...],axis=0)
        cneter_line = self.Line(top_center,bot_center)

        # 分割左右轮廓
        if self.cnt_type == 'step':
            # exclude_ids = top_ids[1:-1]+bot_ids[1:-1]
            exclude_ids = []
            if len(top_ids) == 1 and len(bot_ids) > 1:
                exclude_ids += bot_ids[1:-1]
            elif len(top_ids) > 1 and len(bot_ids) == 1:
                exclude_ids += top_ids[1:-1]
            elif len(top_ids) == 1 and len(bot_ids) == 1:
                exclude_ids += []
            else:
                exclude_ids = top_ids[1:-1]+bot_ids[1:-1]
            lef_cnt,lef_ids = [],[]
            rig_cnt,rig_ids = [],[]
            for i in range(self.pts_num):
                if i in exclude_ids:
                    continue
                x,y = self.contour[i]
                if (i in top_ids and len(top_ids)==1) or (i in bot_ids and len(bot_ids)==1):
                    lef_cnt.append([x,y])
                    lef_ids.append(i)
                    rig_cnt.append([x,y])
                    rig_ids.append(i)
                else:
                    if cneter_line.judgeLR(x,y):
                        lef_cnt.append([x,y])
                        lef_ids.append(i)
                    else:
                        rig_cnt.append([x,y])
                        rig_ids.append(i)
            lef_cnt = np.array(lef_cnt)
            rig_cnt = np.array(rig_cnt)
            lef_ids,lef_cnt = self.reorder(lef_ids,lef_cnt)
            rig_ids,rig_cnt = self.reorder(rig_ids,rig_cnt)
        else:
            top_id = self.update_id(top_id,self.contour)
            bot_id = self.update_id(bot_id,self.contour)
            max_id = np.max([top_id,bot_id])
            min_id = np.min([top_id,bot_id])
            cnt1_id = list(range(min_id,max_id+1))
            cnt2_id = list(range(max_id,self.pts_num))+list(range(0,min_id+1))
            cnt1 = self.contour[cnt1_id,...]
            cnt2 = self.contour[cnt2_id,...]
            
            if self.judgeLR(cnt1[-1],cnt1[-2],cnt2[1]):
                lef_cnt,rig_cnt = cnt1,cnt2
                lef_ids,rig_ids = cnt1_id,cnt2_id
            else:
                lef_cnt,rig_cnt = cnt2,cnt1
                lef_ids,rig_ids = cnt2_id,cnt1_id
            lef_cnt,rig_cnt = lef_cnt.tolist(),rig_cnt.tolist()
            if self.clock_wise:
                if lef_cnt[0][1]<lef_cnt[-1][1]:
                    lef_cnt = list(reversed(lef_cnt))
                if rig_cnt[0][1]<rig_cnt[-1][1]:
                    rig_cnt = list(reversed(rig_cnt))
            else:
                if lef_cnt[0][1]>lef_cnt[-1][1]:
                    lef_cnt = list(reversed(lef_cnt))
                if rig_cnt[0][1]>rig_cnt[-1][1]:
                    rig_cnt = list(reversed(rig_cnt))
            lef_cnt,rig_cnt = np.array(lef_cnt),np.array(rig_cnt)
        return lef_cnt,lef_ids, rig_cnt,rig_ids, top_cnt,top_ids, bot_cnt,bot_ids

    def update_id(self, idx, contour, thres=10):
        new_idx = idx
        x,y = contour[idx,:]
        indices = np.argwhere(np.abs(contour[:,1]-y)<thres).reshape(-1)
        if len(indices)>1:
            pts = contour[indices,:]
            if self.cnt_type == 'left baffle':
                new_idx = np.argmax(pts[:,0])
            else:
                new_idx = np.argmin(pts[:,0])
            new_idx = indices[new_idx]
        return new_idx

    def judgeLR(self,pt0,pt1,pt2):
        line1 = self.Line(pt0,pt1)
        line2 = self.Line(pt0,pt2)
        if line1.k is None and line2.k is not None:
            lr = False if pt1[0]>pt0[0] else True
        elif line1.k is not None and line2.k is None:
            lr = True if pt2[0]>pt0[0] else False
        elif line1.k is None and line2.k is None:
            lr = False if pt1[0]>pt2[0] else True
        else:
            if pt1[1]<pt0[1] and pt2[1]<pt0[1]:
                if line1.k * line2.k<0:
                    lr = True if line1.k>line2.k else False
                else:
                    lr = False if line1.k>line2.k else True
            else:
                if line1.k * line2.k<0:
                    lr = True if line1.k<line2.k else False
                else:
                    lr = False if line1.k<line2.k else True
        return lr

    def search_pt(self,contour,pt_id,k0=0.577):
        '''沿着一点搜索与之满足给定斜率的平行点'''
        start_pt = contour[pt_id]
        pts_num = len(contour)
        range_num = len(contour)-1
        if self.judge_clockwise(contour,pt_id):
            # 判断轮廓点的顺序是顺时针还是逆时针
            left_range = self.cycle_range(pt_id-1,range_num,reverse=True,lenght=pts_num)
            right_range = self.cycle_range(pt_id+1,range_num,reverse=False,lenght=pts_num)
        else:
            left_range = self.cycle_range(pt_id+1,range_num,reverse=False,lenght=pts_num)
            right_range = self.cycle_range(pt_id-1,range_num,reverse=True,lenght=pts_num)
        
        # 往给定点左边搜索
        pts_l = []
        pts_l_ids = []
        prev_k = None
        for i in left_range:
            k = self.compute_slope(start_pt, contour[i])
            if prev_k is None: 
                prev_k = k 
            if k is not None and abs(k) <= k0 and abs(k-prev_k) <= 0.267:
                pts_l.append(contour[i])
                pts_l_ids.append(i)
            else:
                break
            prev_k = k

        # 往给定点右边搜索
        pts_r = []
        pts_r_ids = []
        prev_k = None
        for i in right_range:
            k = self.compute_slope(start_pt, contour[i])
            if prev_k is None: 
                prev_k = k 
            if k is not None and abs(k) <= k0 and abs(k-prev_k) <= 0.267:
                pts_r.append(contour[i])
                pts_r_ids.append(i)
            else:
                break
               
        if len(pts_l) != 0 and len(pts_r) == 0:
            pts = list(reversed([start_pt] + pts_l))
            ids = list(reversed([pt_id] + pts_l_ids))
        elif len(pts_l) == 0 and len(pts_r) != 0:
            pts = [start_pt] + pts_r
            ids = [pt_id] + pts_r_ids
        elif len(pts_l) != 0 and len(pts_r) != 0:
            pts = list(reversed([start_pt] + pts_l)) + pts_r
            ids = list(reversed([pt_id] + pts_l_ids)) + pts_r_ids
        else:
            pts = [start_pt]
            ids = [pt_id]
        return pts,ids

    def judge_clockwise(self,contour,pt_id):
        start_pt = contour[pt_id]
        left_pt = contour[pt_id-1]
        right_pt = contour[pt_id+1 if pt_id+1<len(contour) else pt_id+1-len(contour)]
        if left_pt[0] < right_pt[0]:
            return True
        else:
            return False

    def reorder(self,order,cnt):
        diff = np.array(order[1:])-np.array(order[:-1])
        invalid = False
        for i,v in enumerate(diff):
            if abs(v) != 1 and abs(v) != self.pts_num-1:
                invalid = True
                break
        if invalid:
            new_order = order[i+1:] + order[:i+1]
            new_cnt = np.concatenate([cnt[i+1:,:],cnt[:i+1,:]],axis=0)
        else:
            new_order = order
            new_cnt = cnt
        return new_order,new_cnt
        
    def cycle_range(self,start,num,reverse=False,lenght=None):
        lenght = self.pts_num if lenght is None else lenght
        assert num <= lenght, 'num of range should less than max length'
        if reverse:
            r = list(range(start,start-num,-1))
        else:
            r = list(range(start,start+num))
            r = [i if i<lenght else i-lenght for i in r]
        return r

    def compute_slope(self,pt1,pt2):
        if pt1[0]-pt2[0] == 0:
            slope = None
        else:
            slope = (pt1[1]-pt2[1])/(pt1[0]-pt2[0])
        return slope

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
    def dis2d(pt1,pt2):
        return np.sqrt((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)
    cross_pts = []
    cross_dis = []
    for i,pt in enumerate(zip(*line(*pt1, *pt2))):
        pt = (int(pt[0]),int(pt[1]))
        distance = cv2.pointPolygonTest(contour, pt, True)
        # print(distance)
        if abs(distance) < 1:  # 若点在轮廓上
            cross_pts.append(pt)
            cross_dis.append(abs(distance))
    ids = np.argsort(cross_dis)
    if len(ids) == 0:
        return np.asarray(cross_pts).reshape(-1,2)
    pt1 = cross_pts[ids[0]]
    dis = 0
    n = 1
    while(dis<10 and n<len(ids)):
        pt2 = cross_pts[ids[n]]
        dis = dis2d(pt1,pt2)
        n += 1
    idx = [ids[0],ids[n-1]]
    cross_pts = np.array(cross_pts)[idx,...].reshape(-1,2)     
    if len(cross_pts)>=2:
        max_id = np.argmax(cross_pts[...,0])
        min_id = np.argmin(cross_pts[...,0])
        cross_pts = cross_pts[[min_id,max_id],...]
    return cross_pts

def get_larger_step(area_cnts:dict):
    assert sum(key in area_cnts.keys() for key in classes[:3]),f'some key not found'
    
    baffle_l = ContourParse(np.array(area_cnts[classes[0]]),clock_wise=True,cnt_type='left baffle')
    baffle_r = ContourParse(np.array(area_cnts[classes[1]]),clock_wise=False,cnt_type='right baffle')
    step = ContourParse(np.array(area_cnts[classes[2]]),clock_wise=True,cnt_type='step')

    cnts_l = np.vstack((baffle_l.right_cnt, step.top_cnt, baffle_r.left_cnt, step.bot_cnt))
    cnts_xl = np.vstack((baffle_l.left_cnt, step.top_cnt, baffle_r.right_cnt, step.bot_cnt))
    
    # img = visual_json(data=area_cnts)
    # img = visual_contour(img,baffle_r.left_cnt,draw_line=True)
    # img = visual_contour(img,cnts_l,color=(127,127,127))
    # img = visual_contour(img,cnts_xl,color=(200,200,200))
    # cv2.imshow('img',img)
    # cv2.waitKey(0)
    return cnts_l,cnts_xl


def get_larger_step_v2(area_cnts:dict):
    assert sum(key in area_cnts.keys() for key in classes[:3]),f'some key not found'
    baffle_l = ContourParse(np.array(area_cnts[classes[0]]),clock_wise=True,cnt_type='baffle')
    baffle_r = ContourParse(np.array(area_cnts[classes[1]]),clock_wise=False,cnt_type='baffle')
    step = ContourParse(np.array(area_cnts[classes[2]]),clock_wise=True,cnt_type='step')

    cnts_l = step.contour.copy()
    cnts_xl = step.contour.copy()

    min_x = np.min(baffle_l.contour[...,0])-5
    max_x = np.max(baffle_l.contour[...,0])+5
    for pt,idx in zip(step.left_cnt,step.left_ids):
        pt_tmp = (max(pt[0],max_x),pt[1])
        cross_pts = get_cross_pts(baffle_l.contour,pt_tmp,(min_x,pt[1]))
        if len(cross_pts)>0:
            pt_l,pt_r = cross_pts[0],cross_pts[1]
            cnts_l[idx,:] = pt_r
            cnts_xl[idx,:] = pt_l

    min_x = np.min(baffle_r.contour[...,0])-5 
    max_x = np.max(baffle_r.contour[...,0])+5
    for pt,idx in zip(step.right_cnt,step.right_ids):
        pt_tmp = (min(pt[0],min_x),pt[1])
        cross_pts = get_cross_pts(baffle_r.contour,pt_tmp,(max_x,pt[1]))
        if len(cross_pts)>0:
            pt_l,pt_r = cross_pts[0],cross_pts[1]
            cnts_l[idx,:] = pt_l
            cnts_xl[idx,:] = pt_r


    # img = visual_json(data=area_cnts)
    # img = visual_contour(img,cnts_l,color=(127,127,127))
    # img = visual_contour(img,cnts_xl,color=(200,200,200))
    # cv2.imwrite('out.jpg',img)

    return cnts_l,cnts_xl

def get_larger_floor(area_cnts:dict,add_pixel=30):
    assert 'floor plate' in area_cnts,f'floor plate not found'
    if "imgHeight" not in area_cnts or "imgWidth" not in area_cnts:
        imgH,imgW = 720,1280
    else:
        imgH,imgW = area_cnts["imgHeight"],area_cnts["imgWidth"]
    floor_plate = ContourParse(np.array(area_cnts[classes[3]]),clock_wise=True,cnt_type='step')
    cnts = floor_plate.contour.copy()
    cnts_l = floor_plate.contour.copy()
    x_min,x_max = np.min(cnts[:,0]),np.max(cnts[:,0])
    y_min,y_max = np.min(cnts[:,1]),np.max(cnts[:,1])
    supply = int((y_max-y_min)/10)
    moments = cv2.moments(cnts)             # 求矩
    cx = int(moments['m10']/moments['m00']) # 求x坐标
    cy = int(moments['m01']/moments['m00']) # 求y坐标
    for i,pt in enumerate(cnts):
        x,y = pt[0],pt[1]
        if y>=cy+supply:
            cnts_l[i] = np.asarray([x,np.clip(y+add_pixel,0,imgH-1)])
    
    # img = visual_json(data=area_cnts)
    # img = visual_contour(img,cnts_l,color=(200,200,200))
    # img = cv2.circle(img ,(cx,cy+supply),2,(0,0,255),4) #画出重心
    # cv2.imshow('img',img)
    # cv2.waitKey(0)

    return cnts_l


def loadJson(file):
    assert os.path.exists(file),f'file not exist: `{file}`'
    with open(file,'r',encoding='utf-8') as f:
        data = json.load(f)
    return data

def visual_json(img=None,data=None,file=None):
    assert data is not None or file is not None
    if data is None:
        data = loadJson(file)
    try:
        h,w = data['imgHeight'],data['imgWidth']
    except:
        h,w = 720,1280
    if img is None:
        mask = np.zeros((h,w,3))
    else:
        mask = img.copy()
    for i,(key,value) in enumerate(data.items()):
        if key not in classes:
            continue
        cnt = np.array(data[key])
        visual_mask = cv2.fillPoly(mask,[cnt],color=SEG_COLORS[i])
    visual_mask = visual_mask.astype(np.uint8)
    # cv2.imshow('img',visual_mask)
    # cv2.waitKey(0)
    return visual_mask

def visual_contour(img,cnt,color=(127,127,127),draw_line=False):
    if not draw_line:
        cv2.drawContours(img,[cnt],-1,color,3)
    else:
        for (x1,y1),(x2,y2) in zip(cnt[1:,:],cnt[:-1,:]):
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2), int(y2))
            cv2.line(img,pt1,pt2,color,3,cv2.LINE_AA)
    return img

def line(r0, c0, r1, c1):
    """Generate line pixel coordinates.

    Parameters
    ----------
    r0, c0 : int
        Starting position (row, column).
    r1, c1 : int
        End position (row, column).

    Returns
    -------
    rr, cc : (N,) ndarray of int
        Indices of pixels that belong to the line.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.

    Notes
    -----
    Anti-aliased line generator is available with `line_aa`.

    Examples
    --------
    >>> from skimage.draw import line
    >>> img = np.zeros((10, 10), dtype=np.uint8)
    >>> rr, cc = line(1, 1, 8, 8)
    >>> img[rr, cc] = 1
    >>> img
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    """

    steep = 0
    r = r0
    c = c0
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)

    rr = np.zeros(max(dc, dr) + 1, dtype=np.intp)
    cc = np.zeros(max(dc, dr) + 1, dtype=np.intp)

    sc = 1 if (c1 - c) > 0 else -1
    sr = 1 if (r1 - r) > 0 else -1
    
    if dr > dc:
        steep = 1
        c, r = r, c
        dc, dr = dr, dc
        sc, sr = sr, sc
    d = (2 * dr) - dc

    for i in range(dc):
        if steep:
            rr[i] = c
            cc[i] = r
        else:
            rr[i] = r
            cc[i] = c
        while d >= 0:
            r = r + sr
            d = d - (2 * dc)
        c = c + sc
        d = d + (2 * dr)

    rr[dc] = r1
    cc[dc] = c1

    return np.asarray(rr), np.asarray(cc)


if __name__ == "__main__":
    get_larger_step_v2(loadJson('output/seg_result.json'))