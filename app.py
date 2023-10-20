from src.AIModules.FastSamModule import FastSamModule
from src.AIModules.YoloV8Module import YoloV8Module

from sensor_msgs.msg import CompressedImage
import gradio as gr
import numpy as np
import rospy
import cv2

ai_module_changing = True
ai_module = None

rgb_image = np.zeros((430, 640, 3), dtype=np.uint8)
depth_image = np.zeros((430, 640), dtype=np.uint8)


def rgb_callback(msg):
    global rgb_image
    np_arr = np.frombuffer(msg.data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    rgb_image = img.copy()


def depth_callback(msg):
    global depth_image
    np_arr = np.frombuffer(msg.data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    depth_image = img.copy()


def img_rgb_handler() -> np.ndarray:
    global rgb_image
    return cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)


def img_depth_handler() -> np.ndarray:
    global depth_image
    return depth_image


def process_image_handler() -> np.ndarray:
    global ai_module, ai_module_changing, rgb_image, depth_image
    if ai_module_changing:
        return np.zeros((430, 640, 3), dtype=np.uint8)

    results = ai_module.process(rgb_image)
    processed_image = ai_module.draw_results(rgb_image, results)
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    return processed_image


def btn1_handler() -> None:
    global ai_module, ai_module_changing
    if not isinstance(ai_module, FastSamModule):
        ai_module_changing = True
        if ai_module is not None:
            ai_module.deinitiate()
        ai_module = FastSamModule()
        ai_module.initiate()
        ai_module_changing = False


def btn2_handler() -> None:
    global ai_module, ai_module_changing
    if not isinstance(ai_module, YoloV8Module):
        ai_module_changing = True
        if ai_module is not None:
            ai_module.deinitiate()
        ai_module = YoloV8Module()
        ai_module.initiate()
        ai_module_changing = False


with gr.Blocks() as demo:
    gr.Markdown("<h1 align='center'>AIMSM - AI Models Switching Mechanism</h1>")
    with gr.Row():
        img_rgb = gr.Image(
            img_rgb_handler, min_width=300, show_label=False, every=1 / 60
        )
        img_depth = gr.Image(
            img_depth_handler, min_width=300, show_label=False, every=1 / 60
        )
        processed_image = gr.Image(
            process_image_handler, min_width=300, show_label=False, every=1 / 20
        )
    with gr.Row():
        btn1 = gr.Button("Semantic Mapping")
        btn2 = gr.Button("Object Detection")

    btn1.click(btn1_handler)
    btn2.click(btn2_handler)


if __name__ == "__main__":
    rospy.init_node("my_node_name")
    rospy.Subscriber(
        "/RobotAtVirtualHome/VirtualCameraRGB", CompressedImage, rgb_callback
    )
    rospy.Subscriber(
        "/RobotAtVirtualHome/VirtualCameraDepth", CompressedImage, depth_callback
    )

    demo.queue(api_open=False)
    demo.launch()
