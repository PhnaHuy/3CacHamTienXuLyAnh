"""Display, rotate, and detect faces in video and webcam feeds."""

import base64
import os
import re
import shutil
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import imutils
import numpy as np
import pytesseract


class VideoRotationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Rotation")
        self.root.geometry("1200x760")
        self.root.minsize(900, 560)
        self.colors = {
            "window": "#F3F5F8",
            "sidebar": "#FFFFFF",
            "display": "#111827",
            "text": "#172033",
            "muted": "#687386",
            "accent": "#2563EB",
            "accent_hover": "#1D4ED8",
            "secondary": "#E9EEF5",
            "secondary_hover": "#DCE4EF",
            "border": "#E2E8F0",
        }
        self.root.configure(bg=self.colors["window"])

        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(
            "TNotebook",
            background=self.colors["window"],
            borderwidth=0,
            tabmargins=(8, 8, 0, 0),
        )
        style.configure("TFrame", background=self.colors["window"])
        style.configure(
            "TNotebook.Tab",
            background=self.colors["secondary"],
            foreground=self.colors["muted"],
            borderwidth=0,
            padding=(18, 10),
            font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", self.colors["sidebar"])],
            foreground=[("selected", self.colors["accent"])],
        )

        self.capture = None
        self.rotation = 0
        self.held_keys = set()
        self.key_release_jobs = {}
        self.display_image = None

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.video_tab = ttk.Frame(self.notebook)
        self.faces_tab = ttk.Frame(self.notebook)
        self.contours_tab = ttk.Frame(self.notebook)
        self.plate_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.video_tab, text="Video Rotation")
        self.notebook.add(self.faces_tab, text="Face Detection")
        self.notebook.add(self.contours_tab, text="Object Counting")
        self.notebook.add(self.plate_tab, text="License Plate OCR")
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

        self.face_capture = None
        self.face_frame = None
        self.face_boxes = ()
        self.face_display_image = None
        self.contour_source_image = None
        self.contour_display_image = None
        self.plate_source_image = None
        self.plate_characters = []
        self.plate_display_image = None
        self.plate_display_photo = None
        self.tesseract_path = self.find_tesseract()
        if self.tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
        self.face_detector = None
        if hasattr(cv2, "CascadeClassifier"):
            classifier_path = os.path.join(
                cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
            )
            if os.path.isfile(classifier_path):
                detector = cv2.CascadeClassifier(classifier_path)
                if not detector.empty():
                    self.face_detector = detector

        def create_page(tab, title, description):
            tab.grid_columnconfigure(0, weight=0, minsize=210)
            tab.grid_columnconfigure(1, weight=1)
            tab.grid_rowconfigure(0, weight=1)

            sidebar = tk.Frame(
                tab,
                bg=self.colors["sidebar"],
                width=210,
                padx=18,
                pady=20,
                highlightthickness=1,
                highlightbackground=self.colors["border"],
            )
            sidebar.grid(row=0, column=0, sticky="nsew")
            sidebar.grid_propagate(False)
            tk.Label(
                sidebar,
                text=title,
                bg=self.colors["sidebar"],
                fg=self.colors["text"],
                font=("Segoe UI", 15, "bold"),
                anchor=tk.W,
                wraplength=172,
                justify=tk.LEFT,
            ).pack(fill=tk.X, pady=(0, 6))
            tk.Label(
                sidebar,
                text=description,
                bg=self.colors["sidebar"],
                fg=self.colors["muted"],
                font=("Segoe UI", 9),
                anchor=tk.W,
                justify=tk.LEFT,
                wraplength=172,
            ).pack(fill=tk.X, pady=(0, 20))

            display = tk.Frame(tab, bg=self.colors["window"], padx=12, pady=12)
            display.grid(row=0, column=1, sticky="nsew")
            return sidebar, display

        def add_button(parent, text, command, primary=False):
            background = (
                self.colors["accent"] if primary else self.colors["secondary"]
            )
            foreground = "#FFFFFF" if primary else self.colors["text"]
            hover_background = (
                self.colors["accent_hover"]
                if primary
                else self.colors["secondary_hover"]
            )
            button = tk.Button(
                parent,
                text=text,
                command=command,
                anchor=tk.W,
                bg=background,
                fg=foreground,
                activebackground=hover_background,
                activeforeground=foreground,
                disabledforeground=self.colors["muted"],
                relief=tk.FLAT,
                bd=0,
                padx=12,
                pady=10,
                cursor="hand2",
                font=("Segoe UI", 9, "bold"),
                highlightthickness=0,
            )
            button.pack(fill=tk.X, pady=4)
            return button

        def add_section_label(parent, text):
            tk.Label(
                parent,
                text=text,
                bg=self.colors["sidebar"],
                fg=self.colors["muted"],
                font=("Segoe UI", 9, "bold"),
                anchor=tk.W,
            ).pack(fill=tk.X, pady=(12, 5))

        def create_view(parent, text):
            view = tk.Label(
                parent,
                text=text,
                bg=self.colors["display"],
                fg="#E5E7EB",
                font=("Segoe UI", 12),
                justify=tk.CENTER,
                anchor=tk.CENTER,
            )
            view.pack(fill=tk.BOTH, expand=True)
            return view

        video_sidebar, video_display = create_page(
            self.video_tab,
            "Video Rotation",
            "Choose a video source and rotate the image while it plays.",
        )
        add_section_label(video_sidebar, "VIDEO SOURCE")
        add_button(video_sidebar, "Open Video File", self.open_video, primary=True)
        add_button(video_sidebar, "Open Webcam", self.open_webcam)
        add_button(video_sidebar, "Stop Video", self.stop_video)
        add_section_label(video_sidebar, "KEYBOARD SHORTCUTS")
        tk.Label(
            video_sidebar,
            text="A   Rotate left 90°\nD   Rotate right 90°\nQ / Esc   Stop",
            bg=self.colors["sidebar"],
            fg=self.colors["muted"],
            font=("Segoe UI", 9),
            anchor=tk.W,
            justify=tk.LEFT,
        ).pack(fill=tk.X)
        self.status = tk.StringVar(value="Select a video source.")
        tk.Label(
            video_sidebar,
            textvariable=self.status,
            bg=self.colors["sidebar"],
            fg=self.colors["muted"],
            font=("Segoe UI", 9),
            wraplength=172,
            justify=tk.LEFT,
            anchor=tk.W,
        ).pack(side=tk.BOTTOM, fill=tk.X, pady=(16, 0))
        self.video_view = create_view(
            video_display,
            "Select a video file or open the webcam to start.",
        )

        face_sidebar, face_display = create_page(
            self.faces_tab,
            "Face Detection",
            "Detect and save face regions from a live webcam.",
        )
        add_section_label(face_sidebar, "WEBCAM")
        add_button(
            face_sidebar,
            "Start Face Detection",
            self.open_face_webcam,
            primary=True,
        )
        add_button(face_sidebar, "Stop Webcam", self.stop_face_webcam)
        add_section_label(face_sidebar, "DETECTED FACES")
        self.face_count = tk.StringVar(value="Faces detected: 0")
        tk.Label(
            face_sidebar,
            textvariable=self.face_count,
            bg=self.colors["sidebar"],
            fg=self.colors["accent"],
            font=("Segoe UI", 12, "bold"),
            anchor=tk.W,
        ).pack(fill=tk.X, pady=(2, 8))
        tk.Label(
            face_sidebar,
            text="Press Ctrl+S to save detected faces as PNG.",
            bg=self.colors["sidebar"],
            fg=self.colors["muted"],
            font=("Segoe UI", 9),
            wraplength=172,
            justify=tk.LEFT,
            anchor=tk.W,
        ).pack(fill=tk.X)
        self.face_view = create_view(
            face_display, "Open the webcam to detect faces."
        )

        contour_sidebar, contour_display = create_page(
            self.contours_tab,
            "Object Counting",
            "Find object contours and draw them on the source image.",
        )
        add_section_label(contour_sidebar, "IMAGE")
        add_button(
            contour_sidebar, "Open Image", self.open_contour_image, primary=True
        )
        add_section_label(contour_sidebar, "FILTER")
        tk.Label(
            contour_sidebar,
            text="Minimum contour area",
            bg=self.colors["sidebar"],
            fg=self.colors["muted"],
            font=("Segoe UI", 9),
            anchor=tk.W,
        ).pack(fill=tk.X, pady=(0, 5))
        self.minimum_contour_area = tk.StringVar(value="100")
        tk.Spinbox(
            contour_sidebar,
            from_=0,
            to=1000000,
            increment=50,
            textvariable=self.minimum_contour_area,
            relief=tk.FLAT,
            bd=0,
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            highlightcolor=self.colors["accent"],
            bg="#F8FAFC",
            fg=self.colors["text"],
            buttonbackground=self.colors["secondary"],
            font=("Segoe UI", 10),
            width=12,
        ).pack(fill=tk.X, pady=(0, 8), ipady=6)
        add_button(
            contour_sidebar, "Detect Objects", self.detect_objects, primary=True
        )
        self.object_count = tk.StringVar(value="Objects detected: 0")
        tk.Label(
            contour_sidebar,
            textvariable=self.object_count,
            bg=self.colors["sidebar"],
            fg=self.colors["accent"],
            font=("Segoe UI", 11, "bold"),
            wraplength=172,
            justify=tk.LEFT,
            anchor=tk.W,
        ).pack(fill=tk.X, pady=(16, 0))
        self.contour_view = create_view(
            contour_display, "Open an image to count objects using contours."
        )
        self.contour_view.bind("<Configure>", self.refresh_contour_view)

        plate_sidebar, plate_display = create_page(
            self.plate_tab,
            "License Plate OCR",
            "Recognize plate text and export individual character images.",
        )
        add_section_label(plate_sidebar, "PLATE IMAGE")
        add_button(
            plate_sidebar, "Open Plate Image", self.open_plate_image, primary=True
        )
        add_button(plate_sidebar, "Recognize & Segment", self.process_plate)
        add_button(
            plate_sidebar, "Save Character PNGs", self.save_plate_characters
        )
        add_section_label(plate_sidebar, "OCR TEXT")
        self.plate_text = tk.StringVar()
        self.plate_text_entry = tk.Entry(
            plate_sidebar,
            textvariable=self.plate_text,
            state="readonly",
            relief=tk.FLAT,
            bd=0,
            readonlybackground="#F1F5F9",
            fg=self.colors["text"],
            font=("Segoe UI", 11, "bold"),
        )
        self.plate_text_entry.pack(fill=tk.X, ipady=8, pady=(0, 4))
        add_button(plate_sidebar, "Copy Text", self.copy_plate_text)
        add_button(plate_sidebar, "Set Tesseract Path", self.select_tesseract)
        engine_status = (
            f"Tesseract ready: {os.path.basename(self.tesseract_path)}"
            if self.tesseract_path
            else "Tesseract not found. Set its executable path."
        )
        self.plate_status = tk.StringVar(
            value=f"Open a plate image. {engine_status}"
        )
        tk.Label(
            plate_sidebar,
            textvariable=self.plate_status,
            bg=self.colors["sidebar"],
            fg=self.colors["muted"],
            font=("Segoe UI", 9),
            wraplength=172,
            justify=tk.LEFT,
            anchor=tk.W,
        ).pack(side=tk.BOTTOM, fill=tk.X, pady=(16, 0))
        self.plate_view = create_view(
            plate_display, "Open an image containing a license plate."
        )
        self.plate_view.bind("<Configure>", self.refresh_plate_view)

        self.root.bind("<KeyPress>", self.on_key_press)
        self.root.bind("<KeyRelease>", self.on_key_release)
        self.root.bind_all("<Control-s>", self.save_faces)
        self.root.bind_all("<Control-S>", self.save_faces)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(15, self.update_frame)
        self.root.after(30, self.update_face_frame)

    def open_video(self):
        video_path = filedialog.askopenfilename(
            parent=self.root,
            title="Select a video file",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.mpeg *.mpg"),
                ("All files", "*.*"),
            ],
        )
        if not video_path:
            return

        candidate = cv2.VideoCapture(video_path)
        if not candidate.isOpened():
            candidate.release()
            messagebox.showerror(
                "Unable to Open Video",
                "The selected video could not be opened.",
                parent=self.root,
            )
            return

        self.start_video(candidate, "Playing video")

    def open_webcam(self):
        candidate = cv2.VideoCapture(0)
        if not candidate.isOpened():
            candidate.release()
            messagebox.showerror(
                "Unable to Open Webcam",
                "The webcam could not be opened. Check that it is connected and available.",
                parent=self.root,
            )
            return

        self.start_video(candidate, "Webcam is running")

    def start_video(self, capture, status):
        self.stop_video()
        self.capture = capture
        self.rotation = 0
        self.status.set(status)

    def stop_video(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.rotation = 0
        self.display_image = None
        self.video_view.configure(
            image="",
            text=(
                "Select a video file or open the webcam to start.\n"
                "A: Rotate counter-clockwise | D: Rotate clockwise | Q/Esc: Stop"
            ),
        )
        self.video_view.image = None
        self.status.set("No video is running.")

    def open_face_webcam(self):
        if self.face_detector is None:
            messagebox.showerror(
                "Face Detection Unavailable",
                "This OpenCV installation does not include the Haar cascade "
                "face detector. Install the project dependencies with "
                "'python -m pip install -r requirements.txt'.",
                parent=self.root,
            )
            return

        self.stop_video()
        self.stop_face_webcam()
        candidate = cv2.VideoCapture(0)
        if not candidate.isOpened():
            candidate.release()
            messagebox.showerror(
                "Unable to Open Webcam",
                "The webcam could not be opened. Check that it is connected and available.",
                parent=self.root,
            )
            return

        self.face_capture = candidate
        self.face_view.configure(image="", text="")
        self.face_view.image = None
        self.face_count.set("Faces detected: 0")

    def stop_face_webcam(self):
        if self.face_capture is not None:
            self.face_capture.release()
            self.face_capture = None
        self.face_frame = None
        self.face_boxes = ()
        self.face_view.configure(image="", text="Open the webcam to detect faces.")
        self.face_view.image = None
        self.face_display_image = None
        self.face_count.set("Faces detected: 0")

    def on_tab_changed(self, _event):
        if self.notebook.select() != str(self.faces_tab):
            self.stop_face_webcam()

    def open_contour_image(self):
        image_path = filedialog.askopenfilename(
            parent=self.root,
            title="Select an image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not image_path:
            return

        image = cv2.imread(image_path)
        if image is None:
            messagebox.showerror(
                "Unable to Open Image",
                "The selected file could not be read as an image.",
                parent=self.root,
            )
            return

        self.contour_source_image = image
        self.detect_objects()

    def detect_objects(self):
        if self.contour_source_image is None:
            messagebox.showinfo(
                "No Image Selected",
                "Open an image before detecting objects.",
                parent=self.root,
            )
            return

        try:
            minimum_area = int(self.minimum_contour_area.get())
        except ValueError:
            messagebox.showerror(
                "Invalid Minimum Area",
                "Minimum contour area must be a non-negative integer.",
                parent=self.root,
            )
            return
        if minimum_area < 0:
            messagebox.showerror(
                "Invalid Minimum Area",
                "Minimum contour area must be a non-negative integer.",
                parent=self.root,
            )
            return

        gray = cv2.cvtColor(self.contour_source_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _hierarchy = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        detected_contours = [
            contour for contour in contours if cv2.contourArea(contour) >= minimum_area
        ]

        self.contour_display_image = self.contour_source_image.copy()
        cv2.drawContours(
            self.contour_display_image,
            detected_contours,
            -1,
            (0, 255, 0),
            2,
        )
        self.object_count.set(f"Objects detected: {len(detected_contours)}")
        self.refresh_contour_view()

    def refresh_contour_view(self, _event=None):
        if self.contour_display_image is not None:
            self.render_frame(
                self.contour_display_image,
                self.contour_view,
                "contour_display_photo",
            )

    @staticmethod
    def find_tesseract():
        configured_path = os.environ.get("TESSERACT_CMD")
        if configured_path and os.path.isfile(configured_path):
            return configured_path

        path_from_environment = shutil.which("tesseract")
        if path_from_environment:
            return path_from_environment

        install_locations = (
            os.path.join(
                os.environ.get("ProgramFiles", r"C:\Program Files"),
                "Tesseract-OCR",
                "tesseract.exe",
            ),
            os.path.join(
                os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                "Tesseract-OCR",
                "tesseract.exe",
            ),
        )
        return next(
            (path for path in install_locations if os.path.isfile(path)), None
        )

    def select_tesseract(self):
        executable = filedialog.askopenfilename(
            parent=self.root,
            title="Select tesseract.exe",
            filetypes=[("Tesseract executable", "tesseract.exe"), ("Executable files", "*.exe")],
        )
        if not executable:
            return

        previous_path = pytesseract.pytesseract.tesseract_cmd
        pytesseract.pytesseract.tesseract_cmd = executable
        try:
            pytesseract.get_tesseract_version()
            languages = pytesseract.get_languages(config="")
        except (pytesseract.TesseractNotFoundError, pytesseract.TesseractError) as error:
            pytesseract.pytesseract.tesseract_cmd = previous_path
            messagebox.showerror(
                "Invalid Tesseract Installation",
                f"Could not use the selected OCR executable:\n{error}",
                parent=self.root,
            )
            return

        if "eng" not in languages:
            pytesseract.pytesseract.tesseract_cmd = previous_path
            messagebox.showerror(
                "English OCR Data Not Found",
                "The selected Tesseract installation does not have the English "
                "language data (eng.traineddata). Install it and try again.",
                parent=self.root,
            )
            return

        self.tesseract_path = executable
        self.plate_status.set(f"Tesseract ready: {executable}")

    def open_plate_image(self):
        image_path = filedialog.askopenfilename(
            parent=self.root,
            title="Select a license plate image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not image_path:
            return

        image = cv2.imread(image_path)
        if image is None:
            messagebox.showerror(
                "Unable to Open Image",
                "The selected file could not be read as an image.",
                parent=self.root,
            )
            return

        self.plate_source_image = image
        self.plate_characters = []
        self.plate_display_image = None
        self.plate_text.set("")
        self.plate_status.set("Image loaded. Select Recognize and Segment to process it.")
        self.refresh_plate_view()

    def process_plate(self):
        if self.plate_source_image is None:
            messagebox.showinfo(
                "No Image Selected",
                "Open a license plate image before running OCR.",
                parent=self.root,
            )
            return

        gray = cv2.cvtColor(self.plate_source_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        ocr_image = cv2.resize(
            gray,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC,
        )
        ocr_image = cv2.threshold(
            ocr_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]
        try:
            recognized_text = pytesseract.image_to_string(
                ocr_image,
                lang="eng",
                config=(
                    "--oem 3 --psm 6 "
                    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                ),
            )
            character_boxes = pytesseract.image_to_boxes(
                ocr_image,
                lang="eng",
                config=(
                    "--oem 3 --psm 6 "
                    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                ),
            )
        except pytesseract.TesseractNotFoundError:
            self.update_plate_preview()
            messagebox.showerror(
                "Tesseract OCR Not Found",
                "Could not find tesseract.exe. Use Set Tesseract Path to select "
                "the installed executable.",
                parent=self.root,
            )
            self.plate_status.set(
                "Tesseract not found. Use Set Tesseract Path to locate tesseract.exe."
            )
            return
        except pytesseract.TesseractError as error:
            messagebox.showerror(
                "OCR Failed",
                "Tesseract could not process this image. Confirm that English "
                "language data (eng.traineddata) is installed.\n\n"
                f"Details:\n{error}",
                parent=self.root,
            )
            self.plate_status.set("OCR failed. Check the Tesseract language data and image.")
            return

        recognized_text = re.sub(r"[^A-Z0-9]", "", recognized_text.upper())
        self.plate_text.set(recognized_text)
        self.plate_characters = self.characters_from_tesseract_boxes(
            character_boxes,
            ocr_image.shape[1],
            ocr_image.shape[0],
            self.plate_source_image,
        )
        if not self.plate_characters:
            self.plate_characters = self.segment_plate_characters(gray)
        self.update_plate_preview()
        if recognized_text:
            self.plate_status.set(
                f"Detected {len(self.plate_characters)} character(s). "
                "Select the OCR result and press Ctrl+C to copy."
            )
        else:
            self.plate_status.set(
                "No text was recognized. Use a clear, close-up image of the plate."
            )
        if recognized_text and not self.plate_characters:
            self.plate_status.set(
                "OCR found text but no character crops. Open a close-up image "
                "containing only the plate and try again."
            )
        self.refresh_plate_view()

    def update_plate_preview(self):
        annotated = self.plate_source_image.copy()
        for x, y, width, height, _character_image in self.plate_characters:
            cv2.rectangle(
                annotated, (x, y), (x + width, y + height), (0, 255, 0), 2
            )
        self.plate_display_image = annotated
        self.refresh_plate_view()

    @staticmethod
    def characters_from_tesseract_boxes(
        box_output, ocr_width, ocr_height, source_image
    ):
        image_height, image_width = source_image.shape[:2]
        scale_x = image_width / ocr_width
        scale_y = image_height / ocr_height
        characters = []
        for line in box_output.splitlines():
            fields = line.split()
            if len(fields) < 5:
                continue
            character = fields[0]
            if (
                len(character) != 1
                or character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            ):
                continue
            try:
                left, bottom, right, top = map(int, fields[1:5])
            except ValueError:
                continue

            x1 = max(0, min(image_width, int(left * scale_x)))
            x2 = max(0, min(image_width, int(right * scale_x)))
            y1 = max(0, min(image_height, int((ocr_height - top) * scale_y)))
            y2 = max(0, min(image_height, int((ocr_height - bottom) * scale_y)))
            if x2 <= x1 or y2 <= y1:
                continue
            padding_x = max(1, (x2 - x1) // 12)
            padding_y = max(1, (y2 - y1) // 12)
            crop_x1 = max(0, x1 - padding_x)
            crop_y1 = max(0, y1 - padding_y)
            crop_x2 = min(image_width, x2 + padding_x)
            crop_y2 = min(image_height, y2 + padding_y)
            crop = source_image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
            if crop.size == 0:
                continue
            characters.append((x1, y1, x2 - x1, y2 - y1, crop))
        return characters

    @staticmethod
    def segment_plate_characters(binary_image):
        if len(binary_image.shape) == 3:
            gray = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = binary_image
        height, width = gray.shape
        _, thresholded = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        border_pixels = cv2.copyMakeBorder(
            thresholded, 1, 1, 1, 1, cv2.BORDER_REPLICATE
        )
        border_values = (
            border_pixels[0, 1:-1],
            border_pixels[-1, 1:-1],
            border_pixels[1:-1, 0],
            border_pixels[1:-1, -1],
        )
        if float(cv2.mean(np.concatenate(border_values))[0]) > 127:
            thresholded = cv2.bitwise_not(thresholded)

        contours, _hierarchy = cv2.findContours(
            thresholded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        candidates = []
        max_char_width = max(12, int(height * 0.35))
        for contour in contours:
            x, y, char_width, char_height = cv2.boundingRect(contour)
            if (
                char_width > max_char_width
                or char_width / max(char_height, 1) > 1.2
                or cv2.contourArea(contour) < 8
            ):
                continue
            padding_x = max(1, int(char_width * 0.08))
            padding_y = max(1, int(char_height * 0.06))
            left = max(0, x - padding_x)
            top = max(0, y - padding_y)
            right = min(width, x + char_width + padding_x)
            bottom = min(height, y + char_height + padding_y)
            candidates.append(
                (
                    x,
                    y,
                    char_width,
                    char_height,
                    thresholded[top:bottom, left:right].copy(),
                )
            )

        if candidates:
            median_height = float(
                np.median([candidate[3] for candidate in candidates])
            )
            min_height = max(6, int(median_height * 0.45))
            candidates = [
                candidate for candidate in candidates if candidate[3] >= min_height
            ]

        candidates = [
            candidate
            for candidate in candidates
            if not any(
                other is not candidate
                and other[0] <= candidate[0]
                and other[1] <= candidate[1]
                and other[0] + other[2] >= candidate[0] + candidate[2]
                and other[1] + other[3] >= candidate[1] + candidate[3]
                and other[2] * other[3] > candidate[2] * candidate[3]
                for other in candidates
            )
        ]
        candidates.sort(key=lambda item: item[1] + item[3] / 2)
        rows = []
        row_tolerance = max(5, int(height * 0.04))
        for candidate in candidates:
            center_y = candidate[1] + candidate[3] / 2
            matching_row = next(
                (
                    row
                    for row in rows
                    if abs(row["center"] - center_y) <= row_tolerance
                ),
                None,
            )
            if matching_row is None:
                rows.append({"center": center_y, "characters": [candidate]})
            else:
                matching_row["characters"].append(candidate)
                matching_row["center"] = sum(
                    item[1] + item[3] / 2
                    for item in matching_row["characters"]
                ) / len(matching_row["characters"])

        ordered = []
        for row in sorted(rows, key=lambda item: item["center"]):
            ordered.extend(sorted(row["characters"], key=lambda item: item[0]))
        return ordered

    def save_plate_characters(self):
        if not self.plate_characters:
            messagebox.showinfo(
                "No Characters to Save",
                "Run recognition and segmentation before saving character images.",
                parent=self.root,
            )
            return

        output_dir = filedialog.askdirectory(
            parent=self.root,
            title="Select a folder to save character images",
        )
        if not output_dir:
            return

        timestamp = str(time.time_ns())
        for index, (_x, _y, _width, _height, character_image) in enumerate(
            self.plate_characters, start=1
        ):
            output_path = os.path.join(
                output_dir, f"plate_char_{timestamp}_{index:02d}.png"
            )
            if not cv2.imwrite(output_path, character_image):
                messagebox.showerror(
                    "Unable to Save Character",
                    f"OpenCV could not save the image:\n{output_path}",
                    parent=self.root,
                )
                return

        messagebox.showinfo(
            "Characters Saved",
            f"Saved {len(self.plate_characters)} character image(s) as PNG.",
            parent=self.root,
        )

    def copy_plate_text(self):
        text = self.plate_text.get()
        if not text:
            messagebox.showinfo(
                "No OCR Text",
                "Run OCR before copying the recognized text.",
                parent=self.root,
            )
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.plate_status.set("OCR result copied to the clipboard.")

    def refresh_plate_view(self, _event=None):
        if self.plate_display_image is not None:
            self.render_frame(
                self.plate_display_image,
                self.plate_view,
                "plate_display_photo",
            )
        elif self.plate_source_image is not None:
            self.render_frame(
                self.plate_source_image,
                self.plate_view,
                "plate_display_photo",
            )

    def save_faces(self, _event=None):
        if self.notebook.select() != str(self.faces_tab):
            return
        if self.face_frame is None or not self.face_boxes:
            messagebox.showinfo(
                "No Faces to Save",
                "No faces are currently detected.",
                parent=self.root,
            )
            return "break"

        output_dir = filedialog.askdirectory(
            parent=self.root,
            title="Select a folder to save detected faces",
        )
        if not output_dir:
            return "break"

        frame = self.face_frame
        saved_paths = []
        timestamp = str(time.time_ns())
        for index, (x, y, width, height) in enumerate(self.face_boxes, start=1):
            face = frame[y : y + height, x : x + width]
            if face.size == 0:
                continue
            output_path = os.path.join(
                output_dir, f"face_{timestamp}_{index:03d}.png"
            )
            if not cv2.imwrite(output_path, face):
                messagebox.showerror(
                    "Unable to Save Face",
                    f"OpenCV could not save the image:\n{output_path}",
                    parent=self.root,
                )
                return "break"
            saved_paths.append(output_path)

        if not saved_paths:
            messagebox.showerror(
                "Unable to Save Faces",
                "No valid face regions were available to save.",
                parent=self.root,
            )
        else:
            messagebox.showinfo(
                "Faces Saved",
                f"Saved {len(saved_paths)} face image(s) as PNG.",
                parent=self.root,
            )
        return "break"

    def on_key_press(self, event):
        key = event.keysym.lower()
        if key in self.key_release_jobs:
            self.root.after_cancel(self.key_release_jobs.pop(key))
            return
        elif key in self.held_keys:
            return
        else:
            self.held_keys.add(key)

        if key == "a":
            self.rotation = (self.rotation + 90) % 360
        elif key == "d":
            self.rotation = (self.rotation - 90) % 360
        elif key in ("q", "escape"):
            if self.notebook.select() == str(self.faces_tab):
                self.stop_face_webcam()
            else:
                self.stop_video()

    def on_key_release(self, event):
        key = event.keysym.lower()
        if key not in self.held_keys or key in self.key_release_jobs:
            return
        self.key_release_jobs[key] = self.root.after(
            100, lambda released_key=key: self.finish_key_release(released_key)
        )

    def finish_key_release(self, key):
        self.key_release_jobs.pop(key, None)
        self.held_keys.discard(key)

    def update_frame(self):
        if self.capture is not None:
            success, frame = self.capture.read()
            if success:
                frame = imutils.rotate_bound(frame, angle=self.rotation)
                self.display_frame(frame)
            else:
                self.stop_video()
                self.status.set("Video ended.")

        self.root.after(15, self.update_frame)

    def update_face_frame(self):
        if self.face_capture is not None:
            success, frame = self.face_capture.read()
            if success:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                )
                self.face_frame = frame.copy()
                self.face_boxes = tuple(
                    tuple(int(value) for value in face) for face in faces
                )
                display_frame = frame.copy()
                for x, y, width, height in self.face_boxes:
                    cv2.rectangle(
                        display_frame,
                        (x, y),
                        (x + width, y + height),
                        (0, 255, 0),
                        2,
                    )
                self.face_count.set(f"Faces detected: {len(self.face_boxes)}")
                self.render_frame(display_frame, self.face_view, "face_display_image")
            else:
                self.stop_face_webcam()

        self.root.after(30, self.update_face_frame)

    def display_frame(self, frame):
        self.render_frame(frame, self.video_view, "display_image")

    def render_frame(self, frame, view, image_attribute):
        view_width = view.winfo_width()
        view_height = view.winfo_height()
        if view_width <= 1 or view_height <= 1:
            return

        height, width = frame.shape[:2]
        scale = min(view_width / width, view_height / height)
        resized = cv2.resize(
            frame,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        )
        success, encoded = cv2.imencode(".png", resized)
        if not success:
            raise RuntimeError("OpenCV could not encode the video frame for display.")

        image_data = base64.b64encode(encoded).decode("ascii")
        image = tk.PhotoImage(
            master=self.root, data=image_data, format="png"
        )
        setattr(self, image_attribute, image)
        view.configure(image=image, text="")
        view.image = image

    def close(self):
        self.stop_video()
        self.stop_face_webcam()
        self.root.destroy()


def main():
    root = tk.Tk()
    VideoRotationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
