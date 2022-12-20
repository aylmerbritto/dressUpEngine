import cv2
import numpy as np
import logging
from gabriel_server import cognitive_engine
from gabriel_protocol import gabriel_pb2
import openrtist_pb2
from maskBG import bgMask
logger = logging.getLogger(__name__)
import json

from dressUpInference import dressUpInference

class DressUpEngine(cognitive_engine.Engine):
    SOURCE_NAME = "openrtist"

    def __init__(self, compression_params):
        self.compression_params = compression_params
        wtr_mrk4 = cv2.imread("../wtrMrk.png", -1)
        self.mrk, _, _, mrk_alpha = cv2.split(wtr_mrk4)
        self.alpha = mrk_alpha.astype(float) / 255
        logger.info("FINISHED-INITIALISATION")
        self.obj = dressUpInference()
        self.mask = bgMask()
    
    def handle(self, input_frame):
        extras = cognitive_engine.unpack_extras(openrtist_pb2.Extras, input_frame)
        clothId = json.loads(extras.style)['clothID']
        np_data = np.frombuffer(input_frame.payloads[0], dtype=np.uint8)
        orig_img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        image = orig_img
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.mask.run(image)
        if image is not None:
            image = self.obj.runInference(image, clothId)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = self._apply_watermark(image)
        else:
            image = orig_img
        img_data = orig_img.tostring()

        result = gabriel_pb2.ResultWrapper.Result()
        result.payload_type = gabriel_pb2.PayloadType.IMAGE
        result.payload = img_data
        _, jpeg_img = cv2.imencode(".jpg", image, self.compression_params)
        img_data = jpeg_img.tostring()

        result = gabriel_pb2.ResultWrapper.Result()
        result.payload_type = gabriel_pb2.PayloadType.IMAGE
        result.payload = img_data

        extras = openrtist_pb2.Extras()

        status = gabriel_pb2.ResultWrapper.Status.SUCCESS
        result_wrapper = cognitive_engine.create_result_wrapper(status)
        result_wrapper.results.append(result)
        result_wrapper.extras.Pack(extras)
        return result_wrapper
    
    def _apply_watermark(self, image):
        img_mrk = image[-30:, -120:]  # The waterMark is of dimension 30x120
        img_mrk[:, :, 0] = (1 - self.alpha) * img_mrk[:, :, 0] + self.alpha * self.mrk
        img_mrk[:, :, 1] = (1 - self.alpha) * img_mrk[:, :, 1] + self.alpha * self.mrk
        img_mrk[:, :, 2] = (1 - self.alpha) * img_mrk[:, :, 2] + self.alpha * self.mrk
        image[-30:, -120:] = img_mrk
        # img_out = image.astype("uint8")
        img_out = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return img_out
    
