
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image, ImageTk
from pathlib import Path
from datetime import datetime
import csv
import winsound
import threading
import time

# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = "models/improved_fire_smoke_cnn.keras"
DATASET_PATH = Path("dataset/FIRE-SMOKE-DATASET/Train")
CSV_PATH = Path("detection_history.csv")

IMAGE_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 70
STABLE_FRAMES = 5

CLASS_NAMES = sorted(
    folder.name for folder in DATASET_PATH.iterdir()
    if folder.is_dir()
)

# ==========================================
# LOAD MODEL
# ==========================================

print("Loading CNN model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully!")


# ==========================================
# DASHBOARD APPLICATION
# ==========================================

class FireSmokeDashboard:

    def __init__(self, root):

        self.root = root
        self.root.title("Fire & Smoke Detection System")
        self.root.geometry("1150x780")
        self.root.minsize(950, 650)
        self.root.configure(bg="#101827")

        self.camera = None
        self.running = False
        self.photo = None

        self.counts = {
            "Fire": 0,
            "Smoke": 0,
            "Neutral": 0
        }

        self.last_logged_class = None
        self.candidate_class = None
        self.candidate_count = 0

        self.history = []

        # Alarm control
        self.alarm_playing = False

        self.create_csv()
        self.create_interface()

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close_application
        )

    # ======================================
    # CSV SETUP
    # ======================================

    def create_csv(self):

        if not CSV_PATH.exists():

            with open(
                CSV_PATH,
                "w",
                newline=""
            ) as file:

                writer = csv.writer(file)

                writer.writerow([
                    "Timestamp",
                    "Prediction",
                    "Confidence"
                ])

    # ======================================
    # CREATE UI
    # ======================================

    def create_interface(self):

        # Header
        header = tk.Frame(
            self.root,
            bg="#172338",
            height=75
        )

        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header,
            text="🔥 FIRE & SMOKE DETECTION",
            font=("Segoe UI", 21, "bold"),
            fg="white",
            bg="#172338"
        ).pack(side="left", padx=25, pady=18)

        self.system_label = tk.Label(
            header,
            text="● SYSTEM READY",
            font=("Segoe UI", 11, "bold"),
            fg="#22c55e",
            bg="#172338"
        )

        self.system_label.pack(
            side="right",
            padx=25
        )

        # Main content
        main = tk.Frame(
            self.root,
            bg="#101827"
        )

        main.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=15
        )

        # Statistics
        stats = tk.Frame(
            main,
            bg="#101827"
        )

        stats.pack(fill="x", pady=(0, 15))

        self.counter_labels = {}

        for name, color in [
            ("Fire", "#ef4444"),
            ("Smoke", "#f59e0b"),
            ("Neutral", "#22c55e")
        ]:

            card = tk.Frame(
                stats,
                bg="#1e293b",
                padx=15,
                pady=12
            )

            card.pack(
                side="left",
                fill="x",
                expand=True,
                padx=6
            )

            tk.Label(
                card,
                text=name.upper(),
                font=("Segoe UI", 11, "bold"),
                fg=color,
                bg="#1e293b"
            ).pack(anchor="w")

            number = tk.Label(
                card,
                text="0",
                font=("Segoe UI", 25, "bold"),
                fg="white",
                bg="#1e293b"
            )

            number.pack(anchor="w")

            self.counter_labels[name] = number

        # Two-column layout
        content = tk.Frame(
            main,
            bg="#101827"
        )

        content.pack(
            fill="both",
            expand=True
        )

        # Camera panel
        left = tk.Frame(
            content,
            bg="#1e293b",
            padx=15,
            pady=15
        )

        left.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(0, 10)
        )

        tk.Label(
            left,
            text="LIVE CAMERA FEED",
            font=("Segoe UI", 13, "bold"),
            fg="white",
            bg="#1e293b"
        ).pack(anchor="w", pady=(0, 10))

        self.camera_label = tk.Label(
            left,
            text="Camera is stopped\n\nClick START CAMERA",
            font=("Segoe UI", 16),
            fg="#94a3b8",
            bg="#0b1220"
        )

        self.camera_label.pack(
            fill="both",
            expand=True
        )

        self.status_label = tk.Label(
            left,
            text="STATUS: WAITING",
            font=("Segoe UI", 15, "bold"),
            fg="#94a3b8",
            bg="#1e293b"
        )

        self.status_label.pack(pady=12)

        self.confidence_label = tk.Label(
            left,
            text="Confidence: --",
            font=("Segoe UI", 11),
            fg="white",
            bg="#1e293b"
        )

        self.confidence_label.pack()

        # Camera buttons
        controls = tk.Frame(
            left,
            bg="#1e293b"
        )

        controls.pack(fill="x", pady=15)

        self.start_button = tk.Button(
            controls,
            text="▶ START CAMERA",
            command=self.start_camera,
            bg="#16a34a",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            relief="flat",
            padx=15,
            pady=10
        )

        self.start_button.pack(
            side="left",
            fill="x",
            expand=True,
            padx=(0, 5)
        )

        self.stop_button = tk.Button(
            controls,
            text="■ STOP CAMERA",
            command=self.stop_camera,
            bg="#dc2626",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            relief="flat",
            padx=15,
            pady=10,
            state="disabled"
        )

        self.stop_button.pack(
            side="left",
            fill="x",
            expand=True,
            padx=(5, 0)
        )

        # History panel
        right = tk.Frame(
            content,
            bg="#1e293b",
            padx=12,
            pady=12,
            width=390
        )

        right.pack(
            side="right",
            fill="both",
            expand=True,
            padx=(10, 0)
        )

        tk.Label(
            right,
            text="DETECTION HISTORY",
            font=("Segoe UI", 13, "bold"),
            fg="white",
            bg="#1e293b"
        ).pack(anchor="w", pady=(0, 10))

        # Table
        table_frame = tk.Frame(
            right,
            bg="#1e293b"
        )

        table_frame.pack(
            fill="both",
            expand=True
        )

        columns = (
            "time",
            "prediction",
            "confidence"
        )

        self.table = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings"
        )

        self.table.heading("time", text="TIME")
        self.table.heading("prediction", text="RESULT")
        self.table.heading("confidence", text="CONFIDENCE")

        self.table.column("time", width=125)
        self.table.column("prediction", width=90)
        self.table.column("confidence", width=100)

        scrollbar = ttk.Scrollbar(
            table_frame,
            orient="vertical",
            command=self.table.yview
        )

        self.table.configure(
            yscrollcommand=scrollbar.set
        )

        self.table.pack(
            side="left",
            fill="both",
            expand=True
        )

        scrollbar.pack(
            side="right",
            fill="y"
        )

        # History buttons
        buttons = tk.Frame(
            right,
            bg="#1e293b"
        )

        buttons.pack(fill="x", pady=(12, 0))

        tk.Button(
            buttons,
            text="EXPORT CSV",
            command=self.export_csv,
            bg="#2563eb",
            fg="white",
            relief="flat",
            padx=12,
            pady=9
        ).pack(
            side="left",
            fill="x",
            expand=True,
            padx=3
        )

        tk.Button(
            buttons,
            text="CLEAR HISTORY",
            command=self.clear_history,
            bg="#475569",
            fg="white",
            relief="flat",
            padx=12,
            pady=9
        ).pack(
            side="left",
            fill="x",
            expand=True,
            padx=3
        )

        # Footer
        tk.Label(
            main,
            text="CNN-Based Fire & Smoke Classification | Experimental Demo",
            font=("Segoe UI", 9),
            fg="#94a3b8",
            bg="#101827"
        ).pack(pady=(12, 0))

    # ======================================
    # FIRE ALARM SOUND
    # ======================================

    def play_fire_alarm(self):

        if self.alarm_playing:
            return

        self.alarm_playing = True

        def alarm():

            try:
                for _ in range(3):
                    winsound.Beep(1200, 350)
                    winsound.Beep(800, 350)
                    time.sleep(0.15)

            except RuntimeError:
                pass

            finally:
                self.alarm_playing = False

        threading.Thread(
            target=alarm,
            daemon=True
        ).start()

    # ======================================
    # START CAMERA
    # ======================================

    def start_camera(self):

        self.camera = cv2.VideoCapture(0)

        if not self.camera.isOpened():

            self.camera.release()
            self.camera = None

            messagebox.showerror(
                "Camera Error",
                "Unable to access laptop webcam."
            )

            return

        self.running = True

        self.start_button.config(
            state="disabled"
        )

        self.stop_button.config(
            state="normal"
        )

        self.system_label.config(
            text="● CAMERA ACTIVE",
            fg="#22c55e"
        )

        self.update_camera()

    # ======================================
    # CAMERA PREDICTION
    # ======================================

    def update_camera(self):

        if not self.running or self.camera is None:
            return

        ret, frame = self.camera.read()

        if not ret:
            self.stop_camera()
            return

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        resized = cv2.resize(
            rgb,
            IMAGE_SIZE
        )

        image_array = np.expand_dims(
            resized.astype("float32"),
            axis=0
        )

        predictions = model.predict(
            image_array,
            verbose=0
        )[0]

        index = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[index]

        confidence = float(
            predictions[index]
        ) * 100

        # Stable prediction check
        if confidence >= CONFIDENCE_THRESHOLD:

            self.update_stable_prediction(
                predicted_class,
                confidence
            )

        else:
            self.candidate_class = None
            self.candidate_count = 0

        # Display frame
        color = {
            "Fire": (255, 55, 55),
            "Smoke": (255, 165, 0),
            "Neutral": (40, 200, 100)
        }.get(predicted_class, (255, 255, 255))

        cv2.putText(
            frame,
            f"{predicted_class}: {confidence:.1f}%",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        image = Image.fromarray(rgb)
        image.thumbnail((600, 430))

        self.photo = ImageTk.PhotoImage(image)

        self.camera_label.config(
            image=self.photo,
            text=""
        )

        self.status_label.config(
            text=f"STATUS: {predicted_class.upper()}",
            fg={
                "Fire": "#ef4444",
                "Smoke": "#f59e0b",
                "Neutral": "#22c55e"
            }.get(predicted_class, "white")
        )

        self.confidence_label.config(
            text=f"Confidence: {confidence:.2f}%"
        )

        self.root.after(
            10,
            self.update_camera
        )

    # ======================================
    # STABLE PREDICTION CHECK
    # ======================================

    def update_stable_prediction(
        self,
        predicted_class,
        confidence
    ):

        if predicted_class == self.candidate_class:
            self.candidate_count += 1

        else:
            self.candidate_class = predicted_class
            self.candidate_count = 1

        if self.candidate_count >= STABLE_FRAMES:

            if predicted_class != self.last_logged_class:

                self.save_detection(
                    predicted_class,
                    confidence
                )

                self.last_logged_class = predicted_class

    # ======================================
    # SAVE DETECTION
    # ======================================

    def save_detection(
        self,
        predicted_class,
        confidence
    ):

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        record = [
            timestamp,
            predicted_class,
            round(confidence, 2)
        ]

        with open(
            CSV_PATH,
            "a",
            newline=""
        ) as file:

            writer = csv.writer(file)
            writer.writerow(record)

        self.history.append(record)

        self.table.insert(
            "",
            "end",
            values=(
                timestamp,
                predicted_class,
                f"{confidence:.2f}%"
            )
        )

        self.counts[predicted_class] += 1

        self.counter_labels[
            predicted_class
        ].config(
            text=str(
                self.counts[predicted_class]
            )
        )

        self.table.yview_moveto(1)

        # Trigger alarm only for stable Fire detection
        if predicted_class == "Fire":
            self.play_fire_alarm()

    # ======================================
    # STOP CAMERA
    # ======================================

    def stop_camera(self):

        self.running = False

        if self.camera is not None:

            self.camera.release()
            self.camera = None

        self.start_button.config(
            state="normal"
        )

        self.stop_button.config(
            state="disabled"
        )

        self.system_label.config(
            text="● SYSTEM READY",
            fg="#22c55e"
        )

        self.camera_label.config(
            image="",
            text="Camera is stopped\n\nClick START CAMERA"
        )

        self.photo = None

        self.status_label.config(
            text="STATUS: WAITING",
            fg="#94a3b8"
        )

        self.confidence_label.config(
            text="Confidence: --"
        )

        self.candidate_class = None
        self.candidate_count = 0
        self.last_logged_class = None

    # ======================================
    # EXPORT CSV
    # ======================================

    def export_csv(self):

        destination = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[
                ("CSV files", "*.csv")
            ],
            initialfile="fire_smoke_detection_history.csv"
        )

        if not destination:
            return

        try:

            with open(
                CSV_PATH,
                "r",
                newline=""
            ) as source, open(
                destination,
                "w",
                newline=""
            ) as target:

                target.write(source.read())

            messagebox.showinfo(
                "Export Successful",
                "Detection history exported successfully!"
            )

        except OSError as error:

            messagebox.showerror(
                "Export Error",
                str(error)
            )

    # ======================================
    # CLEAR HISTORY
    # ======================================

    def clear_history(self):

        confirm = messagebox.askyesno(
            "Confirm",
            "Delete all saved detection history?"
        )

        if not confirm:
            return

        self.history.clear()

        for item in self.table.get_children():
            self.table.delete(item)

        self.counts = {
            "Fire": 0,
            "Smoke": 0,
            "Neutral": 0
        }

        for name in self.counts:
            self.counter_labels[name].config(
                text="0"
            )

        self.last_logged_class = None
        self.candidate_class = None
        self.candidate_count = 0

        with open(
            CSV_PATH,
            "w",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "Timestamp",
                "Prediction",
                "Confidence"
            ])

        messagebox.showinfo(
            "Cleared",
            "Detection history cleared."
        )

    # ======================================
    # CLOSE APPLICATION
    # ======================================

    def close_application(self):

        self.stop_camera()
        self.root.destroy()


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    root = tk.Tk()

    app = FireSmokeDashboard(root)

    root.mainloop()
