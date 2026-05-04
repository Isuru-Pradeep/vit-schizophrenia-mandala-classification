import sys
import threading
import tkinter as tk
from queue import Empty, Queue
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import json
from datetime import datetime

from PIL import Image, ImageTk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from tools.inference import find_latest_checkpoint_source, generate_explanations, predict_image


UI_LOG_LISTENERS = []


def ui_log(message: str):
    print(f"[ui] {message}")
    for listener in UI_LOG_LISTENERS:
        listener(message)


class MandalaDemoApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Mandala Severity Demo")
        self.root.geometry("900x680")

        self.image_path_var = tk.StringVar()
        self.time_var = tk.StringVar()
        self.checkpoint_var = tk.StringVar(value=self._default_checkpoint_source())

        self.preview_label = None
        self.preview_photo = None
        self.result_var = tk.StringVar(value="Select an image and enter a completion time.")
        self.status_var = tk.StringVar(value="Idle")
        self.model_info_var = tk.StringVar(value="Model: -")
        self.checkpoint_info_var = tk.StringVar(value="Checkpoint folder: -")
        self.explainability_info_var = tk.StringVar(value="Explainability: -")
        self.time_hint_var = tk.StringVar(value="Enter completion time in minutes.")
        self.current_image_var = tk.StringVar(value="Current image: -")
        self.current_time_var = tk.StringVar(value="Current time: -")
        self.current_prediction_var = tk.StringVar(value="Current prediction: -")
        self.probability_vars = {}
        self.probability_bars = {}
        self.progress_bar = None
        self.explanation_canvases = {}
        self.explanation_photos = {}
        self.explanation_title_vars = {}
        self.main_canvas = None
        self.worker_queue = Queue()
        self.prediction_running = False
        self.explainability_running = False
        self.log_text = None
        self.run_button = None
        self.explain_button = None
        self.stop_button = None
        self.save_button = None
        self.reset_button = None
        self.image_entry = None
        self.time_entry = None
        self.checkpoint_entry = None
        self.latest_prediction_result = None
        self.latest_explainability_result = None
        self.current_preview_image = None
        self.prediction_cancel_event = threading.Event()
        self.explainability_cancel_event = threading.Event()

        UI_LOG_LISTENERS.append(self._enqueue_log_message)
        self._build_ui()
        self.root.after(100, self._process_worker_queue)

    def _default_checkpoint_source(self) -> str:
        try:
            checkpoint_source = str(find_latest_checkpoint_source(PROJECT_ROOT / "checkpoints"))
            ui_log(f"Default checkpoint source resolved to: {checkpoint_source}")
            return checkpoint_source
        except FileNotFoundError:
            ui_log("No default checkpoint source found.")
            return ""


    def _build_ui(self):
        outer_frame = ttk.Frame(self.root)
        outer_frame.pack(fill="both", expand=True)

        self.main_canvas = tk.Canvas(outer_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer_frame, orient="vertical", command=self.main_canvas.yview)
        self.main_canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        self.main_canvas.pack(side="left", fill="both", expand=True)

        container = ttk.Frame(self.main_canvas, padding=16)
        canvas_window = self.main_canvas.create_window((0, 0), window=container, anchor="nw")

        def on_container_configure(_event):
            self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))

        def on_canvas_configure(event):
            self.main_canvas.itemconfigure(canvas_window, width=event.width)

        container.bind("<Configure>", on_container_configure)
        self.main_canvas.bind("<Configure>", on_canvas_configure)
        self.main_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        title = ttk.Label(
            container,
            text="Mandala Schizophrenia Severity Showcase",
            font=("Segoe UI", 16, "bold"),
        )
        title.pack(anchor="w")

        subtitle = ttk.Label(
            container,
            text="Pick a mandala image, enter the completion time, and run the trained model.",
        )
        subtitle.pack(anchor="w", pady=(4, 16))

        form = ttk.Frame(container)
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)

        ttk.Label(form, text="Image").grid(row=0, column=0, sticky="w", pady=6)
        self.image_entry = ttk.Entry(form, textvariable=self.image_path_var)
        self.image_entry.grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(form, text="Browse", command=self._choose_image).grid(row=0, column=2, padx=(0, 8))

        ttk.Label(form, text="Completion Time").grid(row=1, column=0, sticky="w", pady=6)
        self.time_entry = ttk.Entry(form, textvariable=self.time_var)
        self.time_entry.grid(row=1, column=1, sticky="ew", padx=8)
        ttk.Label(form, text="minutes").grid(row=1, column=2, sticky="w")
        ttk.Label(form, textvariable=self.time_hint_var, foreground="#7a5c00").grid(
            row=2, column=1, sticky="w", padx=8
        )
        self.time_var.trace_add("write", self._on_time_changed)

        ttk.Label(form, text="Checkpoint Folder").grid(row=2, column=0, sticky="w", pady=6)
        self.checkpoint_entry = ttk.Entry(form, textvariable=self.checkpoint_var)
        self.checkpoint_entry.grid(row=3, column=1, sticky="ew", padx=8)
        ttk.Label(form, text="Checkpoint Folder").grid(row=3, column=0, sticky="w", pady=6)
        ttk.Button(form, text="Browse", command=self._choose_checkpoint).grid(row=3, column=2, padx=(0, 8))

        actions = ttk.Frame(container)
        actions.pack(fill="x", pady=(14, 14))
        self.run_button = ttk.Button(actions, text="Run Prediction", command=self._run_prediction)
        self.run_button.pack(side="left", padx=(0, 8))
        self.explain_button = ttk.Button(actions, text="Generate Explainability", command=self._run_explainability)
        self.explain_button.pack(side="left", padx=(0, 8))
        self.stop_button = ttk.Button(actions, text="Stop", command=self._request_stop)
        self.stop_button.pack(side="left", padx=(0, 8))
        self.stop_button.config(state="disabled")
        self.save_button = ttk.Button(actions, text="Save Results", command=self._save_results)
        self.save_button.pack(side="left", padx=(0, 8))
        self.reset_button = ttk.Button(actions, text="Reset", command=self._reset_ui)
        self.reset_button.pack(side="left")

        progress_frame = ttk.Frame(container)
        progress_frame.pack(fill="x", pady=(0, 14))
        ttk.Label(progress_frame, textvariable=self.status_var).pack(anchor="w")
        self.progress_bar = ttk.Progressbar(progress_frame, mode="indeterminate")
        self.progress_bar.pack(fill="x", pady=(6, 0))

        current_task_frame = ttk.LabelFrame(container, text="Current Task", padding=12)
        current_task_frame.pack(fill="x", pady=(0, 14))
        ttk.Label(current_task_frame, textvariable=self.current_image_var, wraplength=820, justify="left").pack(anchor="w")
        ttk.Label(current_task_frame, textvariable=self.current_time_var, wraplength=820, justify="left").pack(anchor="w", pady=(4, 0))
        ttk.Label(current_task_frame, textvariable=self.current_prediction_var, wraplength=820, justify="left").pack(anchor="w", pady=(4, 0))

        body = ttk.Frame(container)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)

        preview_frame = ttk.LabelFrame(body, text="Image Preview", padding=12)
        preview_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        preview_frame.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)

        self.preview_label = ttk.Label(preview_frame, text="No image selected.", anchor="center")
        self.preview_label.grid(row=0, column=0, sticky="nsew")
        self.preview_label.bind("<Button-1>", lambda _event: self._open_image_popup(self.current_preview_image, "Image Preview"))

        results_frame = ttk.LabelFrame(body, text="Model Output", padding=12)
        results_frame.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        ttk.Label(
            results_frame,
            textvariable=self.result_var,
            wraplength=360,
            justify="left",
        ).pack(anchor="w", pady=(0, 12))

        ttk.Label(
            results_frame,
            textvariable=self.model_info_var,
            wraplength=360,
            justify="left",
        ).pack(anchor="w", pady=(0, 4))

        ttk.Label(
            results_frame,
            textvariable=self.checkpoint_info_var,
            wraplength=360,
            justify="left",
        ).pack(anchor="w", pady=(0, 12))

        ttk.Label(
            results_frame,
            textvariable=self.explainability_info_var,
            wraplength=360,
            justify="left",
        ).pack(anchor="w", pady=(0, 12))

        for class_name in ["healthy", "mild", "moderate", "severe"]:
            var = tk.StringVar(value=f"{class_name}: -")
            self.probability_vars[class_name] = var
            row = ttk.Frame(results_frame)
            row.pack(fill="x", pady=4)
            ttk.Label(row, textvariable=var, font=("Consolas", 11), width=20).pack(side="left")
            bar = ttk.Progressbar(row, orient="horizontal", mode="determinate", maximum=100)
            bar.pack(side="left", fill="x", expand=True, padx=(8, 0))
            self.probability_bars[class_name] = bar

        explain_frame = ttk.LabelFrame(container, text="Explainability View", padding=12)
        explain_frame.pack(fill="both", expand=True, pady=(14, 0))

        notebook = ttk.Notebook(explain_frame)
        notebook.pack(fill="both", expand=True)

        method_titles = [
            (
                "saliency",
                "Saliency",
                "Shows which pixels most strongly influence the prediction. Bright regions matter more, but this view can look noisy.",
            ),
            (
                "gradcam",
                "Grad-CAM",
                "Highlights broader image regions that support the prediction. This is often easier to read than raw saliency.",
            ),
            (
                "attention",
                "Attention",
                "Shows how the Vision Transformer distributes attention across image patches when making its decision.",
            ),
            (
                "integrated_gradients",
                "Integrated Gradients",
                "Estimates pixel contribution by comparing the real image to a baseline image. This is often more stable than saliency.",
            ),
            (
                "occlusion",
                "Occlusion",
                "Covers small parts of the image and checks how much the prediction changes. Bigger changes mean that region is important.",
            ),
        ]

        for method_key, method_label, method_description in method_titles:
            tab = ttk.Frame(notebook, padding=10)
            notebook.add(tab, text=method_label)
            tab.columnconfigure(0, weight=1)
            tab.columnconfigure(1, weight=1)
            tab.rowconfigure(1, weight=1)

            description_label = ttk.Label(
                tab,
                text=method_description,
                wraplength=760,
                justify="left",
            )
            description_label.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

            title_var = tk.StringVar(value=f"Explaining class: - | Note: {method_label}")
            ttk.Label(tab, textvariable=title_var, wraplength=760, justify="left").grid(
                row=1, column=0, columnspan=2, sticky="w", pady=(0, 10)
            )
            self.explanation_title_vars[method_key] = title_var

            overlay_label = ttk.Label(tab, text="Overlay will appear here.", anchor="center")
            overlay_label.grid(row=2, column=0, sticky="nsew", padx=(0, 8))
            overlay_label.bind(
                "<Button-1>",
                lambda _event, key=method_key: self._open_explanation_popup(key, "overlay"),
            )

            map_label = ttk.Label(tab, text="Heatmap will appear here.", anchor="center")
            map_label.grid(row=2, column=1, sticky="nsew", padx=(8, 0))
            map_label.bind(
                "<Button-1>",
                lambda _event, key=method_key: self._open_explanation_popup(key, "grayscale"),
            )

            self.explanation_canvases[method_key] = {
                "overlay": overlay_label,
                "grayscale": map_label,
            }

        log_frame = ttk.LabelFrame(container, text="Activity Log", padding=12)
        log_frame.pack(fill="both", expand=True, pady=(14, 0))
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical")
        self.log_text = tk.Text(log_frame, height=10, wrap="word", yscrollcommand=log_scroll.set, state="disabled")
        log_scroll.config(command=self.log_text.yview)
        log_scroll.pack(side="right", fill="y")
        self.log_text.pack(side="left", fill="both", expand=True)

    def _on_mousewheel(self, event):
        if self.main_canvas is not None:
            self.main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


    def _is_any_task_running(self) -> bool:
        return self.prediction_running or self.explainability_running

    def _refresh_busy_state(self, status_message: str | None = None):
        is_busy = self._is_any_task_running()
        self.root.config(cursor="watch" if is_busy else "")
        if is_busy:
            self.progress_bar.start(10)
        else:
            self.progress_bar.stop()
        if status_message is not None:
            self.status_var.set(status_message)

        prediction_button_state = "disabled" if self.prediction_running else "normal"
        explainability_button_state = "disabled" if self.explainability_running else "normal"
        shared_button_state = "disabled" if is_busy else "normal"
        entry_state = "disabled" if is_busy else "normal"
        stop_button_state = "normal" if is_busy else "disabled"

        if self.run_button is not None:
            self.run_button.config(state=prediction_button_state)
        if self.explain_button is not None:
            self.explain_button.config(state=explainability_button_state)
        if self.stop_button is not None:
            self.stop_button.config(state=stop_button_state)
        for widget in [self.save_button, self.reset_button]:
            if widget is not None:
                widget.config(state=shared_button_state)
        for widget in [self.image_entry, self.time_entry, self.checkpoint_entry]:
            if widget is not None:
                widget.config(state=entry_state)

    def _process_worker_queue(self):
        try:
            while True:
                event_type, payload = self.worker_queue.get_nowait()

                if event_type == "progress":
                    self.status_var.set(payload)
                elif event_type == "ui_log":
                    self._append_log(payload)
                elif event_type == "prediction_success":
                    self._handle_prediction_success(payload)
                elif event_type == "prediction_error":
                    self.prediction_running = False
                    self._refresh_busy_state("Prediction failed.")
                    messagebox.showerror("Prediction error", payload)
                elif event_type == "prediction_cancelled":
                    self.prediction_running = False
                    self._refresh_busy_state("Prediction stopped.")
                    ui_log("Prediction cancellation processed by UI.")
                elif event_type == "explainability_success":
                    self._handle_explainability_success(payload)
                elif event_type == "explainability_partial":
                    self._handle_explainability_partial(payload)
                elif event_type == "explainability_error":
                    self.explainability_running = False
                    self._refresh_busy_state("Explainability failed.")
                    messagebox.showerror("Explainability error", payload)
                elif event_type == "explainability_cancelled":
                    self.explainability_running = False
                    self._refresh_busy_state("Explainability stopped.")
                    self.explainability_info_var.set("Explainability: stopped by user.")
                    ui_log("Explainability cancellation processed by UI.")
        except Empty:
            pass
        finally:
            self.root.after(100, self._process_worker_queue)

    def _enqueue_log_message(self, message: str):
        self.worker_queue.put(("ui_log", message))

    def _append_log(self, message: str):
        if self.log_text is None:
            return
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{timestamp}] {message}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _on_time_changed(self, *_args):
        raw_value = self.time_var.get().strip()
        if not raw_value:
            self.time_hint_var.set("Enter completion time in minutes.")
            return
        try:
            value = float(raw_value)
        except ValueError:
            self.time_hint_var.set("Please enter a numeric value in minutes.")
            return
        if value <= 0:
            self.time_hint_var.set("Completion time should be greater than 0 minutes.")
        elif value > 180:
            self.time_hint_var.set("This is an unusually large completion time. Please double-check.")
        else:
            self.time_hint_var.set("Completion time looks valid.")

    def _update_current_task_summary(self, predicted_label: str | None = None):
        image_text = self.image_path_var.get().strip() or "-"
        time_text = self.time_var.get().strip() or "-"
        self.current_image_var.set(f"Current image: {image_text}")
        self.current_time_var.set(f"Current time: {time_text} minutes")
        if predicted_label is None:
            self.current_prediction_var.set("Current prediction: -")
        else:
            self.current_prediction_var.set(f"Current prediction: {predicted_label}")

    def _open_image_popup(self, pil_image: Image.Image | None, title: str):
        if pil_image is None:
            return
        ui_log(f"Opening image popup: title='{title}', size={pil_image.width}x{pil_image.height}")
        popup = tk.Toplevel(self.root)
        popup.title(title)
        popup.geometry("1100x850")

        original_image = pil_image.copy()
        zoom_var = tk.DoubleVar(value=1.0)

        controls = ttk.Frame(popup, padding=(12, 12, 12, 0))
        controls.pack(fill="x")
        ttk.Label(controls, text="Zoom").pack(side="left")

        canvas_frame = ttk.Frame(popup, padding=12)
        canvas_frame.pack(fill="both", expand=True)

        h_scroll = ttk.Scrollbar(canvas_frame, orient="horizontal")
        v_scroll = ttk.Scrollbar(canvas_frame, orient="vertical")
        canvas = tk.Canvas(
            canvas_frame,
            highlightthickness=0,
            xscrollcommand=h_scroll.set,
            yscrollcommand=v_scroll.set,
        )
        h_scroll.config(command=canvas.xview)
        v_scroll.config(command=canvas.yview)
        h_scroll.pack(side="bottom", fill="x")
        v_scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        image_id = canvas.create_image(0, 0, anchor="nw")
        zoom_label = ttk.Label(controls, text="100%")
        zoom_label.pack(side="left", padx=(8, 0))

        def render_image(scale: float):
            width = max(1, int(original_image.width * scale))
            height = max(1, int(original_image.height * scale))
            resized = original_image.resize((width, height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(resized)
            canvas.itemconfigure(image_id, image=photo)
            canvas.image = photo
            canvas.configure(scrollregion=(0, 0, width, height))
            zoom_label.config(text=f"{int(scale * 100)}%")

        def set_zoom(scale: float):
            scale = min(max(scale, 0.1), 6.0)
            zoom_var.set(scale)
            render_image(scale)
            ui_log(f"Popup zoom updated: title='{title}', zoom={scale:.2f}x")

        def zoom_in():
            set_zoom(zoom_var.get() * 1.25)

        def zoom_out():
            set_zoom(zoom_var.get() / 1.25)

        def fit_to_window():
            canvas_width = max(canvas.winfo_width(), 1)
            canvas_height = max(canvas.winfo_height(), 1)
            fit_scale = min(canvas_width / original_image.width, canvas_height / original_image.height)
            set_zoom(min(fit_scale, 1.0))

        ttk.Button(controls, text="Zoom Out", command=zoom_out).pack(side="left", padx=(12, 4))
        ttk.Button(controls, text="Zoom In", command=zoom_in).pack(side="left", padx=4)
        ttk.Button(controls, text="Fit", command=fit_to_window).pack(side="left", padx=4)
        ttk.Button(controls, text="100%", command=lambda: set_zoom(1.0)).pack(side="left", padx=4)

        def on_canvas_configure(_event):
            if not getattr(popup, "_fit_initialized", False):
                popup._fit_initialized = True
                fit_to_window()

        def on_mousewheel(event):
            if event.state & 0x0004:
                if event.delta > 0:
                    zoom_in()
                elif event.delta < 0:
                    zoom_out()
            else:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<Configure>", on_canvas_configure)
        canvas.bind("<MouseWheel>", on_mousewheel)
        popup.protocol("WM_DELETE_WINDOW", lambda: (ui_log(f"Closing image popup: title='{title}'"), popup.destroy()))
        set_zoom(1.0)

    def _open_explanation_popup(self, method_key: str, image_kind: str):
        if (method_key, image_kind) not in self.explanation_photos:
            return
        if self.latest_explainability_result is None:
            return
        method_result = self.latest_explainability_result["explanations"].get(method_key)
        if method_result is None:
            return
        self._open_image_popup(method_result[image_kind], f"{method_key.title()} - {image_kind.title()}")

    def _choose_image(self):
        ui_log("Image browse dialog opened.")
        file_path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        if file_path:
            self.image_path_var.set(file_path)
            ui_log(f"Image selected: {file_path}")
            self._update_preview(file_path)
            self._update_current_task_summary()
        else:
            ui_log("Image selection cancelled.")

    def _choose_checkpoint(self):
        ui_log("Checkpoint folder dialog opened.")
        directory = filedialog.askdirectory(title="Select checkpoint folder")
        if directory:
            self.checkpoint_var.set(directory)
            ui_log(f"Checkpoint folder selected: {directory}")
        else:
            ui_log("Checkpoint folder selection cancelled.")

    def _request_stop(self):
        if not self._is_any_task_running():
            ui_log("Stop request ignored because no task is running.")
            return
        if self.prediction_running:
            self.prediction_cancel_event.set()
        if self.explainability_running:
            self.explainability_cancel_event.set()
        self.status_var.set("Stop requested. Waiting for tasks to stop safely...")
        ui_log(
            f"Stop requested. active_tasks=(prediction={self.prediction_running}, "
            f"explainability={self.explainability_running})"
        )

    def _update_preview(self, image_path: str):
        ui_log(f"Updating preview for image: {image_path}")
        try:
            image = Image.open(image_path).convert("RGB")
            self.current_preview_image = image.copy()
            image.thumbnail((360, 360))
            self.preview_photo = ImageTk.PhotoImage(image)
            self.preview_label.configure(image=self.preview_photo, text="")
            ui_log("Image preview updated successfully.")
        except Exception as exc:
            self.preview_label.configure(image="", text=f"Preview error: {exc}")
            self.preview_photo = None
            ui_log(f"Image preview failed: {exc}")

    def _run_prediction(self):
        if self.prediction_running:
            ui_log("Prediction request ignored because another prediction task is running.")
            return

        image_path = self.image_path_var.get().strip()
        checkpoint_source = self.checkpoint_var.get().strip() or None
        ui_log(
            f"Run Prediction clicked. image_path='{image_path}', checkpoint_source='{checkpoint_source}'"
        )

        if not image_path:
            ui_log("Prediction blocked: no image selected.")
            messagebox.showerror("Missing image", "Please select an image first.")
            return

        try:
            completion_time = float(self.time_var.get().strip())
        except ValueError:
            ui_log(f"Prediction blocked: invalid completion time '{self.time_var.get().strip()}'.")
            messagebox.showerror("Invalid time", "Please enter a numeric completion time.")
            return

        def report_progress(message: str):
            ui_log(f"Prediction progress: {message}")
            self.worker_queue.put(("progress", message))

        def worker():
            try:
                result = predict_image(
                    image_path=image_path,
                    completion_time=completion_time,
                    checkpoint_source=checkpoint_source,
                    progress_callback=report_progress,
                    cancel_callback=self.prediction_cancel_event.is_set,
                )
                self.worker_queue.put(("prediction_success", (image_path, result)))
            except InterruptedError:
                ui_log("Prediction stopped by user.")
                self.worker_queue.put(("prediction_cancelled", None))
            except Exception as exc:
                ui_log(f"Prediction failed: {exc}")
                self.worker_queue.put(("prediction_error", str(exc)))

        ui_log(f"Starting prediction with completion_time={completion_time}.")
        ui_log(
            f"Prediction task scheduled. concurrent_state="
            f"(prediction_running=False->True, explainability_running={self.explainability_running})"
        )
        self._update_current_task_summary()
        self.prediction_cancel_event = threading.Event()
        self.prediction_running = True
        self._refresh_busy_state("Starting prediction...")
        threading.Thread(target=worker, daemon=True).start()

    def _handle_prediction_success(self, payload):
        image_path, result = payload
        self.prediction_running = False
        self._refresh_busy_state("Prediction complete.")
        ui_log("Prediction UI cleanup complete.")
        self.latest_prediction_result = result
        self._update_preview(image_path)
        self.result_var.set(
            f"Predicted severity: {result['predicted_label']}\n"
            f"Used {result['num_checkpoints']} checkpoint(s) on {result['device']}."
        )
        self.model_info_var.set(f"Model: {result.get('model_name', '-')}")
        self.checkpoint_info_var.set(f"Checkpoint folder: {result.get('checkpoint_dir', '-')}")
        ui_log(
            f"Prediction complete. label={result['predicted_label']}, "
            f"model={result.get('model_name', '-')}, device={result['device']}"
        )
        self._update_current_task_summary(result["predicted_label"])

        for class_name, var in self.probability_vars.items():
            probability = result["probabilities"].get(class_name)
            if probability is None:
                var.set(f"{class_name}: not used")
                self.probability_bars[class_name]["value"] = 0
            else:
                var.set(f"{class_name:<9} {probability * 100:6.2f}%")
                self.probability_bars[class_name]["value"] = probability * 100
        ui_log(f"Prediction probabilities updated: {result['probabilities']}")

    def _set_explanation_image(self, method_key: str, image_kind: str, pil_image: Image.Image):
        display_image = pil_image.copy()
        display_image.thumbnail((320, 320))
        photo = ImageTk.PhotoImage(display_image)
        self.explanation_photos[(method_key, image_kind)] = photo
        self.explanation_canvases[method_key][image_kind].configure(image=photo, text="")
        ui_log(f"Updated explainability image for method={method_key}, kind={image_kind}.")

    def _reset_explanation_views(self):
        for method_key in self.explanation_canvases:
            self.explanation_canvases[method_key]["overlay"].configure(image="", text="Overlay will appear here.")
            self.explanation_canvases[method_key]["grayscale"].configure(image="", text="Heatmap will appear here.")
            self.explanation_title_vars[method_key].set(
                f"Explaining class: - | Note: {method_key.replace('_', ' ').title()}"
            )
            self.explanation_photos.pop((method_key, "overlay"), None)
            self.explanation_photos.pop((method_key, "grayscale"), None)

    def _handle_explainability_partial(self, payload):
        method_key, method_result, metadata = payload
        if self.latest_explainability_result is None:
            self.latest_explainability_result = {
                "explanations": {},
                "errors": {},
            }

        self.latest_explainability_result.update(metadata)
        self.latest_explainability_result["explanations"][method_key] = method_result
        self.latest_explainability_result["errors"].pop(method_key, None)

        self._set_explanation_image(method_key, "overlay", method_result["overlay"])
        self._set_explanation_image(method_key, "grayscale", method_result["grayscale"])

        target_name = metadata.get("target_class_name", "-")
        self.explanation_title_vars[method_key].set(
            f"Explaining class: {target_name} | Click an image to enlarge"
        )
        self.model_info_var.set(f"Model: {metadata.get('model_name', '-')}")
        checkpoint_dir = metadata.get("checkpoint_dir", metadata.get("checkpoint_path", "-"))
        num_checkpoints = metadata.get("num_checkpoints", 1)
        self.checkpoint_info_var.set(
            f"Checkpoint folder: {checkpoint_dir} ({num_checkpoints} checkpoint(s) used for explainability)"
        )

        completed_methods = len(self.latest_explainability_result["explanations"])
        total_methods = len(self.explanation_canvases)
        self.explainability_info_var.set(
            f"Explainability: {completed_methods}/{total_methods} method(s) ready."
        )
        ui_log(f"Displayed explainability output for method={method_key}.")

    def _run_explainability(self):
        if self.explainability_running:
            ui_log("Explainability request ignored because another explainability task is running.")
            return

        image_path = self.image_path_var.get().strip()
        checkpoint_source = self.checkpoint_var.get().strip() or None
        ui_log(
            f"Generate Explainability clicked. image_path='{image_path}', checkpoint_source='{checkpoint_source}'"
        )

        if not image_path:
            ui_log("Explainability blocked: no image selected.")
            messagebox.showerror("Missing image", "Please select an image first.")
            return

        try:
            completion_time = float(self.time_var.get().strip())
        except ValueError:
            ui_log(f"Explainability blocked: invalid completion time '{self.time_var.get().strip()}'.")
            messagebox.showerror("Invalid time", "Please enter a numeric completion time.")
            return

        def report_progress(message: str):
            ui_log(f"Explainability progress: {message}")
            self.worker_queue.put(("progress", message))

        def report_result(method_key: str, method_result: dict, metadata: dict):
            self.worker_queue.put(("explainability_partial", (method_key, method_result, metadata)))

        def worker():
            try:
                result = generate_explanations(
                    image_path=image_path,
                    completion_time=completion_time,
                    checkpoint_source=checkpoint_source,
                    progress_callback=report_progress,
                    result_callback=report_result,
                    cancel_callback=self.explainability_cancel_event.is_set,
                )
                self.worker_queue.put(("explainability_success", result))
            except InterruptedError:
                ui_log("Explainability stopped by user.")
                self.worker_queue.put(("explainability_cancelled", None))
            except Exception as exc:
                ui_log(f"Explainability failed: {exc}")
                self.worker_queue.put(("explainability_error", str(exc)))

        ui_log(f"Starting explainability with completion_time={completion_time}.")
        ui_log(
            f"Explainability task scheduled. concurrent_state="
            f"(prediction_running={self.prediction_running}, explainability_running=False->True)"
        )
        self._update_current_task_summary(
            self.latest_prediction_result["predicted_label"] if self.latest_prediction_result else None
        )
        self.latest_explainability_result = {
            "model_name": "-",
            "checkpoint_path": "-",
            "checkpoint_dir": checkpoint_source or "-",
            "num_checkpoints": 0,
            "target_class": None,
            "target_class_name": "-",
            "image_size": None,
            "explanations": {},
            "errors": {},
        }
        self._reset_explanation_views()
        self.explainability_info_var.set("Explainability: generation started, waiting for first method...")
        self.explainability_cancel_event = threading.Event()
        self.explainability_running = True
        self._refresh_busy_state("Starting explainability...")
        threading.Thread(target=worker, daemon=True).start()

    def _handle_explainability_success(self, result):
        self.explainability_running = False
        self._refresh_busy_state("Explainability complete.")
        ui_log("Explainability UI cleanup complete.")
        self.latest_explainability_result = result

        for method_key in self.explanation_canvases:
            if method_key not in result["explanations"]:
                self.explanation_canvases[method_key]["overlay"].configure(
                    image="",
                    text="No output for this method.",
                )
                self.explanation_canvases[method_key]["grayscale"].configure(
                    image="",
                    text="No output for this method.",
                )
                self.explanation_photos.pop((method_key, "overlay"), None)
                self.explanation_photos.pop((method_key, "grayscale"), None)
                ui_log(f"Cleared explainability outputs for failed method={method_key}.")
                self.explanation_title_vars[method_key].set(f"Explaining class: - | Method output unavailable")

        for method_key, method_result in result["explanations"].items():
            self._set_explanation_image(method_key, "overlay", method_result["overlay"])
            self._set_explanation_image(method_key, "grayscale", method_result["grayscale"])
            target_name = result.get("target_class_name", "-")
            self.explanation_title_vars[method_key].set(
                f"Explaining class: {target_name} | Click an image to enlarge"
            )

        self.model_info_var.set(f"Model: {result.get('model_name', '-')}")
        checkpoint_dir = result.get("checkpoint_dir", result.get("checkpoint_path", "-"))
        num_checkpoints = result.get("num_checkpoints", 1)
        self.checkpoint_info_var.set(
            f"Checkpoint folder: {checkpoint_dir} ({num_checkpoints} checkpoint(s) used for explainability)"
        )
        if result.get("errors"):
            failed_methods = ", ".join(sorted(result["errors"].keys()))
            self.explainability_info_var.set(f"Explainability: partial success. Failed: {failed_methods}")
            ui_log(f"Explainability partial success. Failed methods: {result['errors']}")
        else:
            self.explainability_info_var.set("Explainability: all methods generated successfully.")
            ui_log("Explainability completed successfully for all methods.")
        ui_log(
            "Explainability final result summary: "
            f"successful_methods={sorted(result['explanations'].keys())}, "
            f"failed_methods={sorted(result.get('errors', {}).keys())}"
        )

    def _save_results(self):
        if self.latest_prediction_result is None and self.latest_explainability_result is None:
            messagebox.showinfo("Nothing to save", "Run prediction or explainability first.")
            return

        output_dir = filedialog.askdirectory(title="Select folder to save results")
        if not output_dir:
            ui_log("Save Results cancelled.")
            return

        output_path = Path(output_dir) / f"demo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_path.mkdir(parents=True, exist_ok=True)

        if self.current_preview_image is not None:
            self.current_preview_image.save(output_path / "input_image.png")

        if self.latest_prediction_result is not None:
            with open(output_path / "prediction_summary.json", "w", encoding="utf-8") as file:
                json.dump(self.latest_prediction_result, file, indent=2)

        if self.latest_explainability_result is not None:
            explain_summary = {
                key: value
                for key, value in self.latest_explainability_result.items()
                if key != "explanations"
            }
            with open(output_path / "explainability_summary.json", "w", encoding="utf-8") as file:
                json.dump(explain_summary, file, indent=2)

            for method_key, method_result in self.latest_explainability_result["explanations"].items():
                method_result["overlay"].save(output_path / f"{method_key}_overlay.png")
                method_result["grayscale"].save(output_path / f"{method_key}_grayscale.png")

        ui_log(f"Results saved to: {output_path}")
        messagebox.showinfo("Saved", f"Results saved to:\n{output_path}")

    def _reset_ui(self):
        if self._is_any_task_running():
            ui_log("Reset ignored because a task is running.")
            return

        self.image_path_var.set("")
        self.time_var.set("")
        self.checkpoint_var.set(self._default_checkpoint_source())
        self.result_var.set("Select an image and enter a completion time.")
        self.status_var.set("Idle")
        self.model_info_var.set("Model: -")
        self.checkpoint_info_var.set("Checkpoint folder: -")
        self.explainability_info_var.set("Explainability: -")
        self.current_image_var.set("Current image: -")
        self.current_time_var.set("Current time: -")
        self.current_prediction_var.set("Current prediction: -")
        self.preview_label.configure(image="", text="No image selected.")
        self.preview_photo = None
        self.current_preview_image = None
        self.latest_prediction_result = None
        self.latest_explainability_result = None

        for class_name, var in self.probability_vars.items():
            var.set(f"{class_name}: -")
            self.probability_bars[class_name]["value"] = 0

        self._reset_explanation_views()

        if self.log_text is not None:
            self.log_text.configure(state="normal")
            self.log_text.delete("1.0", "end")
            self.log_text.configure(state="disabled")
        ui_log("UI reset complete.")


def main():
    ui_log("Launching Mandala demo UI.")
    root = tk.Tk()
    app = MandalaDemoApp(root)
    ui_log("Mandala demo UI ready.")
    root.mainloop()


if __name__ == "__main__":
    main()
