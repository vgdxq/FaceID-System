import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
from deepface import DeepFace
import pickle
import os
import time
import sys
from PIL import Image, ImageTk

# Перевірка версії Python
if sys.version_info < (3, 6):
    raise RuntimeError("This application requires Python 3.6 or higher")

class FacialRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Recognition System (DeepFace)")
        self.root.geometry("600x500")

        # База даних користувачів та збереження зображень
        self.db_file = "users_deepface.pkl"
        self.image_dir = "user_images"
        os.makedirs(self.image_dir, exist_ok=True)
        self.users = self.load_users()

        # Завантаження каскаду Haar
        haarcascade_path = "C:/haarcascade_frontalface_default.xml"
        if not os.path.exists(haarcascade_path):
            messagebox.showerror("Error", f"Cannot find Haar cascade file at {haarcascade_path}")
            return
        
        self.face_cascade = cv2.CascadeClassifier(haarcascade_path)
        if self.face_cascade.empty():
            messagebox.showerror("Error", "Failed to load Haar cascade. Ensure the file is at the correct location.")
            return  # Вихід без закриття `self.root`

        # Налаштування вебкамери
        self.cap = None
        for index in range(3):
            self.cap = cv2.VideoCapture(index)
            if self.cap.isOpened():
                break
        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot access webcam.")
            return  # Не закриваємо `self.root`, просто припиняємо ініціалізацію
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Графічний інтерфейс
        tk.Label(root, text="Username:").pack(pady=10)
        self.username_entry = tk.Entry(root)
        self.username_entry.pack()

        tk.Label(root, text="Password:").pack(pady=10)
        self.password_entry = tk.Entry(root, show="*")
        self.password_entry.pack()

        tk.Button(root, text="Register", command=self.register).pack(pady=20)
        tk.Button(root, text="Authorize", command=self.authorize).pack(pady=10)

        # Відображення зображення користувача
        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

    def load_users(self):
        """Завантажує дані користувачів."""
        if os.path.exists(self.db_file):
            with open(self.db_file, "rb") as f:
                return pickle.load(f)
        return {}

    def save_users(self):
        """Зберігає дані користувачів."""
        with open(self.db_file, "wb") as f:
            pickle.dump(self.users, f)

    def capture_image(self, mode="register"):
        """Захоплення зображення обличчя користувача."""
        window_name = f"{mode.capitalize()} Face Capture"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 640, 480)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                messagebox.showerror("Error", "Failed to capture image.")
                cv2.destroyAllWindows()
                return None

            faces = self.face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 5)
            face_status = "Face detected" if len(faces) > 0 else "No face detected"
            status_color = (0, 255, 0) if len(faces) > 0 else (0, 0, 255)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(frame, face_status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            cv2.putText(frame, "Press 'c' to capture, 'q' to quit", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(10) & 0xFF

            if key == ord('c'):
                if len(faces) == 0:
                    messagebox.showwarning("Warning", "No face detected.")
                    continue
                image_path = os.path.join(self.image_dir, f"{mode}_{int(time.time())}.jpg")
                cv2.imwrite(image_path, frame)
                cv2.destroyAllWindows()
                return image_path
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return None

    def register(self):
        """Реєстрація нового користувача."""
        username = self.username_entry.get().strip()
        password = self.password_entry.get().strip()

        if not username or not password:
            messagebox.showerror("Error", "Username and password are required")
            return

        if username in self.users:
            messagebox.showerror("Error", "Username already exists")
            return

        image_path = self.capture_image(mode="register")
        if image_path is None:
            return

        self.users[username] = {"password": password, "image_path": image_path}
        self.save_users()
        messagebox.showinfo("Success", "Registration successful")

    def find_user_by_face(self, new_image_path):
        """Шукає користувача за збігом обличчя."""
        for username, data in self.users.items():
            stored_image_path = data["image_path"]
            try:
                result = DeepFace.verify(
                    img1_path=stored_image_path,
                    img2_path=new_image_path,
                    model_name="Facenet",
                    detector_backend="opencv",
                    enforce_detection=False
                )
                if result["verified"]:
                    return username
            except Exception as e:
                print(f"Error comparing faces for {username}: {e}")
                continue
        return None

    def authorize(self):
        """Авторизація користувача через розпізнавання обличчя."""
        username = self.username_entry.get().strip()
        password = self.password_entry.get().strip()

        if not username or not password:
            messagebox.showerror("Error", "Username and password are required")
            return

        new_image_path = self.capture_image(mode="authorize")
        if new_image_path is None:
            return

        # Перевіряємо, чи існує введений логін
        if username in self.users:
            # Перевіряємо пароль
            if self.users[username]["password"] != password:
                messagebox.showerror("Error", "Invalid password")
                return

            # Перевіряємо обличчя для введеного логіна
            stored_image_path = self.users[username]["image_path"]
            result = DeepFace.verify(
                img1_path=stored_image_path,
                img2_path=new_image_path,
                model_name="Facenet",
                detector_backend="opencv",
                enforce_detection=False
            )

            if result["verified"]:
                messagebox.showinfo("Success", "Access granted")
                self.show_user_image(stored_image_path)
            else:
                # Перевіряємо, чи обличчя належить іншому користувачу
                matched_username = self.find_user_by_face(new_image_path)
                if matched_username and matched_username != username:
                    messagebox.showerror("Error", "User exists in the database, but not under this username")
                else:
                    messagebox.showerror("Error", "Face not found. Please register.")
            os.remove(new_image_path)  # Видаляємо тимчасове зображення
            return

        # Якщо логін не існує, перевіряємо обличчя
        matched_username = self.find_user_by_face(new_image_path)
        if matched_username:
            messagebox.showerror("Error", "User exists in the database, but not under this username")
        else:
            messagebox.showerror("Error", "Face not found. Please register.")
        os.remove(new_image_path)  # Видаляємо тимчасове зображення

    def show_user_image(self, image_path):
        """Відображення зображення користувача."""
        image = Image.open(image_path)
        image = image.resize((200, 200), Image.LANCZOS)
        self.image_label.image = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.image_label.image)

if __name__ == "__main__":
    root = tk.Tk()
    app = FacialRecognitionApp(root)
    root.mainloop()