import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class ImageCropperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Çoklu Kırpma Aracı")

        self.image_path = None
        self.image = None
        self.crop_boxes = []
        self.save_folder = None
        self.rectangles = []  # Kırpma kutularını çizmek için

        # Resim Seçme Butonu
        self.btn_select_image = tk.Button(root, text="Resim Seç", command=self.load_image)
        self.btn_select_image.pack()

        # Kırpma Alanları Listesi
        self.listbox = tk.Listbox(root, height=10)
        self.listbox.pack()

        # Kaydetme Klasörü Seç
        self.btn_select_folder = tk.Button(root, text="Kaydetme Klasörü Seç", command=self.select_save_folder)
        self.btn_select_folder.pack()

        # Kırp ve Kaydet Butonu
        self.btn_crop_save = tk.Button(root, text="Kırp ve Kaydet", command=self.crop_and_save)
        self.btn_crop_save.pack()

        # Resim Alanı
        self.canvas = tk.Canvas(root, width=500, height=500, bg="gray")
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Resim Dosyaları", "*.jpg;*.png;*.jpeg")])
        if file_path:
            self.image_path = file_path
            self.image = Image.open(file_path)
            self.show_image()

    def show_image(self):
        self.canvas.delete("all")
        img_resized = self.image.resize((500, 500))  # Ekrana sığdır
        self.tk_image = ImageTk.PhotoImage(img_resized)
        self.canvas.create_image(250, 250, image=self.tk_image)

    def select_save_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.save_folder = folder_selected
            messagebox.showinfo("Kayıt Klasörü Seçildi", f"Kırpılan resimler burada kaydedilecek: {self.save_folder}")

    def on_press(self, event):
        if len(self.crop_boxes) >= 32:
            messagebox.showwarning("Limit Aşıldı", "En fazla 32 kırpma alanı ekleyebilirsiniz!")
            return
        self.start_x = event.x
        self.start_y = event.y

    def on_release(self, event):
        end_x = event.x
        end_y = event.y
        self.crop_boxes.append((self.start_x, self.start_y, end_x, end_y))
        self.rectangles.append(self.canvas.create_rectangle(self.start_x, self.start_y, end_x, end_y, outline="red"))

        self.listbox.insert(tk.END, f"Kırpma {len(self.crop_boxes)}: {self.start_x}, {self.start_y}, {end_x}, {end_y}")

    def crop_and_save(self):
        if not self.image:
            messagebox.showerror("Hata", "Önce bir resim seçmelisiniz!")
            return

        if not self.save_folder:
            messagebox.showerror("Hata", "Lütfen bir kayıt klasörü seçin!")
            return

        img_width, img_height = self.image.size
        scale_x = img_width / 500  # Orijinal boyuta ölçekleme
        scale_y = img_height / 500

        for i, (x1, y1, x2, y2) in enumerate(self.crop_boxes):
            # Orijinal resimdeki koordinatlara ölçekleme
            real_x1 = int(x1 * scale_x)
            real_y1 = int(y1 * scale_y)
            real_x2 = int(x2 * scale_x)
            real_y2 = int(y2 * scale_y)

            cropped_img = self.image.crop((real_x1, real_y1, real_x2, real_y2))
            cropped_img.save(f"{self.save_folder}/crop_{i+1}.png")

        messagebox.showinfo("Başarılı", "Kırpma işlemi tamamlandı!")

# Uygulama Çalıştırma
root = tk.Tk()
app = ImageCropperApp(root)
root.mainloop()
