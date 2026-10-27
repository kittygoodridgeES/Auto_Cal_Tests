"""BSD 2-Clause License

Copyright (c) 2023, Allied Vision Technologies GmbH
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import sys
from typing import Optional
from queue import Queue, Empty, Full

from vmbpy import *
import ast
from pathlib import Path

# All frames will either be recorded in this format, or transformed to it before being displayed
opencv_display_format = PixelFormat.Bgr8
BASE_DIR = Path(__file__).resolve().parent


class CameraSetup:
    def __init__(self, logger, cutter_no):
        self.logger = logger
        self.cutter_no = cutter_no
        self.settings_path = BASE_DIR / "camera-settings.txt"

    def print_preamble(self ):
        self.logger.info('///////////////////////////////////////////////////')
        self.logger.info('/// VmbPy Asynchronous Grab with OpenCV Example ///')
        self.logger.info('///////////////////////////////////////////////////\n')


    def print_usage(self ):
        self.logger.info('Usage:')
        self.logger.info('    python asynchronous_grab_opencv.py [camera_id]')
        self.logger.info('    python asynchronous_grab_opencv.py [/h] [-h]')
        self.logger.info('Parameters:')
        self.logger.info('    camera_id   ID of the camera to use (using first camera if not specified)')


    def abort(self, reason: str, return_code: int = 1, usage: bool = False):
        self.logger.info(reason + '\n')

        if usage:
            self.print_usage()

        sys.exit(return_code)


    def parse_args(self) -> Optional[str]:
        args = sys.argv[1:]
        args = []
        argc = len(args)

        for arg in args:
            if arg in ('/h', '-h'):
                self.print_usage()
                sys.exit(0)

        if argc > 1:
            self.logger.info("Invalid number of arguments. Abort.")
            self.abort(reason="Invalid number of arguments. Abort.", return_code=2, usage=True)

        return None if argc == 0 else args[0]


    def get_camera(self, camera_id: Optional[str]) -> Camera:
        with VmbSystem.get_instance() as vmb:
            if camera_id:
                try:
                    return vmb.get_camera_by_id(camera_id)

                except VmbCameraError:
                    self.logger.info("Failed to access camera. Abort.")
                    self.abort('Failed to access Camera \'{}\'. Abort.'.format(camera_id))

            else:
                cams = vmb.get_all_cameras()
                if not cams:
                    self.logger.info("No cameras accessible. Abort.")
                    self.abort('No Cameras accessible. Abort.')
                
                return cams[0]


    def setup_camera(self, cam: Camera):
        #read camera settings from camera-settings.txt
        if self.cutter_no.startswith("2."):
            line_number = 11 + 4*(int(self.cutter_no[2:3])-1) + int(self.cutter_no[4:5])
            self.logger.info(f"{line_number} : line number")
        elif self.cutter_no.startswith("3."):
            line_number = 11 + 16 + 4*(int(self.cutter_no[2:3])-1) + int(self.cutter_no[4:5])
            self.logger.info(f"{line_number} : line number")
        else:
            line_number = int(self.cutter_no) -1
            self.logger.info(f"{line_number} : line number")

        with open(self.settings_path, encoding="utf-8") as f:
            lines = f.readlines()

        settings_txt =lines[line_number].rstrip("\n")
        settings_dict = ast.literal_eval(settings_txt)
        self.logger.info(f"cutter number: {settings_dict['cutter_no']}, cutter_no: {self.cutter_no}")
        #check cutter number matches camera settings
        if str(settings_dict["cutter_no"]) != self.cutter_no:
            self.logger.info("Cutter number does not match camera settings. Abort.")
            self.abort("Cutter number does not match camera settings. Abort.")
        else:
            with cam:
                # Enable auto exposure time setting if camera supports it
                try:
                    cam.ExposureAuto.set('Off')
                except (AttributeError, VmbFeatureError):
                    pass

                # Enable auto exposure time setting if camera supports it
                try:
                    cam.ExposureMode.set('Timed')
                except (AttributeError, VmbFeatureError):
                    pass

                # Enable auto exposure time setting if camera supports it
                try:
                    cam.ExposureTime.set(settings_dict["ExposureTime"])
                    self.logger.info('Exposure time set to 2000 µs')
                except (AttributeError, VmbFeatureError):
                    pass

                # Enable white balancing if camera supports it
                try:
                    cam.BalanceWhiteAuto.set('Continuous')
                except (AttributeError, VmbFeatureError):
                    pass

                # Make sure gain is at 0
                try:
                    cam.GainAuto.set('Off')
                except (AttributeError, VmbFeatureError):
                    pass

                # Increase Frame acquisition
                try:
                    cam.AcquisitionFrameRateEnable.set('True')
                    cam.AquisitionFrameRate.set('30')
                except (AttributeError, VmbFeatureError):
                    pass

                try:
                    gamma = cam.get_feature_by_name("Gamma")
                    gamma.set(settings_dict["Gamma"])
                except (AttributeError, VmbFeatureError):
                    self.logger.exception("Error setting Gamma")
                    pass



    def setup_pixel_format(self, cam: Camera):
        # Query available pixel formats. Prefer color formats over monochrome formats
        cam_formats = cam.get_pixel_formats()

        cam_mono_formats = intersect_pixel_formats(cam_formats, MONO_PIXEL_FORMATS)
        convertible_mono_formats = tuple(f for f in cam_mono_formats
                                        if opencv_display_format in f.get_convertible_formats())

        # if OpenCV compatible color format is supported directly, use that
        if opencv_display_format in cam_formats:
            cam.set_pixel_format(opencv_display_format)

        # fall back to a mono format that can be converted
        elif convertible_mono_formats:
            cam.set_pixel_format(convertible_mono_formats[0])

        else:
            self.abort('Camera does not support an OpenCV compatible format. Abort.')


class Handler:
    def __init__(self, logger):
        self.logger = logger
        self.display_queue = Queue(maxsize=10)

    def get_image(self, timeout=2.0):
        """
        Return the next complete camera image.

        Incomplete frames are ignored by the callback. If no complete frame
        reaches the queue before timeout expires, queue.Empty is raised.
        """
        return self.display_queue.get(timeout=timeout)

    def __call__(self, cam: Camera, stream: Stream, frame: Frame):
        try:
            # Keep the original working behaviour: ignore incomplete frames.
            if frame.get_status() != FrameStatus.Complete:
                self.logger.debug(
                    "Ignoring incomplete camera frame: %s",
                    frame.get_status(),
                )
                return

            if frame.get_pixel_format() == opencv_display_format:
                display = frame
            else:
                display = frame.convert_pixel_format(opencv_display_format)

            image = display.as_opencv_image()

            try:
                # Do not block the VmbPy callback if analysis falls behind.
                self.display_queue.put_nowait(image)
            except Full:
                self.logger.debug(
                    "Camera image queue is full; dropping frame"
                )

        except Exception:
            self.logger.exception(
                "Exception while processing camera frame"
            )

        finally:
            try:
                cam.queue_frame(frame)
            except Exception:
                self.logger.exception(
                    "Failed to requeue camera frame"
                )


class Stream:
    FRAME_TIMEOUT_SECONDS = 2.0

    def __init__(self, logger, cutter_no):
        self.logger = logger
        self.display = []
        self.testvar = 0

        logger.info("init camerasetup")
        self.c = CameraSetup(logger, cutter_no)

        logger.info("printing preamble")
        self.c.print_preamble()

        logger.info("parsing args")
        self.cam_id = self.c.parse_args()

    def stop_stream(self):
        """
        Kept for compatibility with existing callers.

        Normal shutdown is handled inside run() via stop_event.
        """
        self.logger.info(
            "Camera stop requested; active stream will stop via its stop event"
        )

    def run(self, stop_event, frame_callback=None):
        """
        Run camera acquisition.

        Incomplete frames are ignored, matching the original working code.
        If no COMPLETE frame is received for FRAME_TIMEOUT_SECONDS, raise so
        the owning multiprocessing.Process exits non-zero.

        If frame_callback is supplied, each complete image is passed directly
        to it.
        """
        try:
            with VmbSystem.get_instance():
                with self.c.get_camera(self.cam_id) as cam:
                    self.logger.info("Camera opened: %s", cam.get_id())
                    self.c.setup_camera(cam)
                    self.logger.info("Camera setup complete")
                    self.c.setup_pixel_format(cam)
                    self.logger.info("Camera pixel format setup complete")

                    handler = Handler(self.logger)
                    streaming_started = False

                    try:
                        cam.start_streaming(
                            handler=handler,
                            buffer_count=10,
                        )
                        streaming_started = True
                        self.logger.info("Camera streaming started")

                        while not stop_event.is_set():
                            try:
                                img = handler.get_image(
                                    timeout=self.FRAME_TIMEOUT_SECONDS
                                )
                            except Empty as exc:
                                raise RuntimeError(
                                    "No complete camera frame received for "
                                    f"{self.FRAME_TIMEOUT_SECONDS:.1f} seconds"
                                ) from exc

                            if img is None or getattr(img, "size", 0) == 0:
                                raise RuntimeError(
                                    "Camera returned an empty image"
                                )

                            self.display = img

                            if frame_callback is not None:
                                frame_callback(img)

                    finally:
                        if streaming_started:
                            try:
                                cam.stop_streaming()
                                self.logger.info(
                                    "Camera streaming stopped"
                                )
                            except Exception:
                                self.logger.exception(
                                    "Error while stopping camera stream"
                                )

        except Exception:
            self.logger.exception(
                "Camera/vision stream failed"
            )
            raise
