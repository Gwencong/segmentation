#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################
import os
import sys
import cv2
import time
import pyds
import numpy as np
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent.resolve().absolute().__str__()
sys.path.append(ROOT)

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
from utils.utils import is_aarch64, bus_call, make_element, create_source_bin, map_mask_as_display_bgr


MUX_OUTPUT_WIDTH = 1280
MUX_OUTPUT_HEIGHT = 720
INPUT_STREAM = [f"file://{ROOT}/data/segment.mp4"]
DEBUG = False
start_time = time.time()
start_time2 = 0
vid_writer = None


def seg_src_pad_buffer_probe(pad, info, u_data):
    global start_time, start_time2, vid_writer
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            if DEBUG:
                n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                # convert python array into numy array format.
                frame_image = np.array(n_frame, copy=True, order="C")
                # covert the array into cv2 default color format
                frame_image = cv2.cvtColor(frame_image, cv2.COLOR_RGBA2BGR)
                # print(frame_image.shape)
                if vid_writer is None:
                    vid_writer = cv2.VideoWriter(
                        "output/record.avi",
                        cv2.VideoWriter_fourcc("X", "V", "I", "D"),
                        25,
                        (frame_image.shape[1], frame_image.shape[0]),
                    )
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        l_user = frame_meta.frame_user_meta_list
        while l_user is not None:
            try:
                # Note that l_user.data needs a cast to pyds.NvDsUserMeta
                # The casting is done by pyds.NvDsUserMeta.cast()
                # The casting also keeps ownership of the underlying memory
                # in the C code, so the Python garbage collector will leave
                # it alone.
                seg_user_meta = pyds.NvDsUserMeta.cast(l_user.data)
            except StopIteration:
                break
            if seg_user_meta and seg_user_meta.base_meta.meta_type == \
                    pyds.NVDSINFER_SEGMENTATION_META:
                try:
                    # Note that seg_user_meta.user_meta_data needs a cast to
                    # pyds.NvDsInferSegmentationMeta
                    # The casting is done by pyds.NvDsInferSegmentationMeta.cast()
                    # The casting also keeps ownership of the underlying memory
                    # in the C code, so the Python garbage collector will leave
                    # it alone.
                    segmeta = pyds.NvDsInferSegmentationMeta.cast(seg_user_meta.user_meta_data)
                except StopIteration:
                    break
                # Retrieve mask data in the numpy format from segmeta
                # Note that pyds.get_segmentation_masks() expects object of
                # type NvDsInferSegmentationMeta
                if DEBUG:
                    masks = pyds.get_segmentation_masks(segmeta)
                    masks = np.array(masks, copy=True, order='C')
                    mask_image = map_mask_as_display_bgr(masks)
                    img = cv2.addWeighted(frame_image,0.7,mask_image,0.3,0)
                    # cv2.imwrite(folder_name + "/" + str(frame_number) + ".jpg", img)
                    vid_writer.write(img)
            try:
                l_user = l_user.next
            except StopIteration:
                break
        
        cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        CurFPS = 1 / (time.time() - start_time)
        AvgFPS = frame_number / (time.time() - start_time2)
        print(f'{cur_time} Frames={frame_number} FPS={CurFPS:.0f} AvgFPS={AvgFPS:.1f}')

        start_time = time.time()
        if int(start_time2) == 0:
            start_time2 = time.time()

        try:
            l_frame = l_frame.next
        except StopIteration:
            break
    return Gst.PadProbeReturn.OK


def main(args):
    # Check input arguments
    
    global folder_name
    folder_name = './output'
    if os.path.exists(folder_name):
        sys.stdout.write("The output folder %s already exists. "
                         "Please remove it first.\n" % folder_name)
    else:
        os.mkdir(folder_name)

    config_file = f"{ROOT}/deepstream/configs/segmentation_config_semantic.txt"
    # Standard GStreamer initialization
    Gst.init(None)

    # Create gstreamer elements
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = make_element("nvstreammux", "Stream-muxer")
    streammux.set_property("width", MUX_OUTPUT_WIDTH)
    streammux.set_property("height", MUX_OUTPUT_HEIGHT)
    streammux.set_property("batch-size", 1)
    streammux.set_property("batched-push-timeout", 4000)
    streammux.set_property("live-source", 0)  # rtsp
    pipeline.add(streammux)

    number_src = len(INPUT_STREAM)
    for i in range(number_src):
        print("Creating source_bin ", i, " \n ")
        uri_name = INPUT_STREAM[i]
        print(uri_name)

        source_bin = create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)

        padname = "sink_%u" % i
        sinkpad = streammux.get_request_pad(padname)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)

    seg = make_element("nvinfer", "primary-nvinference-engine")
    seg.set_property('config-file-path', config_file)
    pgie_batch_size = seg.get_property("batch-size")
    if pgie_batch_size != number_src:
        print("WARNING: Overriding infer-config batch-size", pgie_batch_size,
              " with number of sources ", number_src,
              " \n")
        seg.set_property("batch-size", number_src)

    nvvidconv = make_element("nvvideoconvert", "convertor")

    caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    filter = make_element("capsfilter", "filter")
    filter.set_property("caps", caps)

    nvsegvisual = make_element("nvsegvisual", "nvsegvisual")

    transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")

    print("Creating EGLSink \n")
    sink = make_element("nveglglessink", "nvvideo-renderer")
    sink.set_property('sync', False)

    nvsegvisual.set_property('batch-size', number_src)
    nvsegvisual.set_property('width', 1280)
    nvsegvisual.set_property('height', 720)
    sink.set_property("qos", 0)
    print("Adding elements to Pipeline \n")

    pipeline.add(seg)
    pipeline.add(nvvidconv)
    pipeline.add(filter)
    pipeline.add(nvsegvisual)
    pipeline.add(sink)
    if is_aarch64():
        pipeline.add(transform)

    # we link the elements together
    # file-source -> jpeg-parser -> nvv4l2-decoder ->
    # nvinfer -> nvsegvisual -> sink
    print("Linking elements in the Pipeline \n")
    
    streammux.link(seg)
    seg.link(nvvidconv)
    nvvidconv.link(filter)
    filter.link(nvsegvisual)
    nvsegvisual.link(transform)
    transform.link(sink)
    
    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # Lets add probe to get informed of the meta data generated, we add probe to
    # the src pad of the inference element
    seg_src_pad = filter.get_static_pad("src")
    if not seg_src_pad:
        sys.stderr.write(" Unable to get src pad \n")
    else:
        seg_src_pad.add_probe(Gst.PadProbeType.BUFFER, seg_src_pad_buffer_probe, 0)
        pass

    # List the sources
    print("Now playing...")
    for i, source in enumerate(args[1:-1]):
        if i != 0:
            print(i, ": ", source)

    print("Starting pipeline \n")
    # start play back and listed to events
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    pipeline.set_state(Gst.State.NULL)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
