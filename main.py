from src.AIModules.FastSamModule import FastSamModule
from src.AIModules.YoloV8Module import YoloV8Module
import cv2
import os

OP_MODE = "Image"  # 'Image' or 'Video'

# ai_module = FastSamModule()
ai_module = YoloV8Module()
ai_module.initiate()


if OP_MODE == "Image":
    image_path = os.path.join("assets", "image.jpg")
    image = cv2.imread(image_path)

    results = ai_module.process(image)
    processed_image = ai_module.draw_results(image, results)
    cv2.imshow("processed", processed_image)
    cv2.imshow("original", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif OP_MODE == "Video":
    cap = cv2.VideoCapture(
        "https://cdn.flowplayer.com/a30bd6bc-f98b-47bc-abf5-97633d4faea0/hls/de3f6ca7-2db3-4689-8160-0f574a5996ad/playlist.m3u8",
    )

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        results = ai_module.process(frame)
        processed_image = ai_module.draw_results(frame, results)

        cv2.imshow("processed", processed_image)
        cv2.imshow("original", frame)
        if cv2.waitKey(16) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
