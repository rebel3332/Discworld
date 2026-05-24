import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import json
import os

class SpriteSelector:
    def __init__(self, root):
        self.root = root
        self.root.title("Sprite Sheet Selector")
        self.root.geometry("1200x800")
        
        # Данные
        self.image = None
        self.photo = None
        self.start_x = None
        self.start_y = None
        self.current_rect = None
        self.rectangles = []  # Сохранённые области
        self.current_index = 0
        
        # Создание интерфейса
        self.create_widgets()
        
    def create_widgets(self):
        # === Панель управления ===
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)
        
        # Кнопки
        ttk.Button(control_frame, text="Открыть изображение", 
                  command=self.open_image).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Сохранить координаты", 
                  command=self.save_coordinates).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Загрузить координаты", 
                  command=self.load_coordinates).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Очистить всё", 
                  command=self.clear_all).pack(side=tk.LEFT, padx=5)
        
        # === Информация о выделении ===
        info_frame = ttk.LabelFrame(control_frame, text="Информация", padding="5")
        info_frame.pack(side=tk.LEFT, padx=20)
        
        self.info_label = ttk.Label(info_frame, text="X: 0, Y: 0, W: 0, H: 0")
        self.info_label.pack()
        
        # === Список сохранённых областей ===
        list_frame = ttk.LabelFrame(control_frame, text="Сохранённые области", padding="5")
        list_frame.pack(side=tk.LEFT, padx=20)
        
        self.rect_listbox = tk.Listbox(list_frame, width=40, height=3)
        self.rect_listbox.pack()
        self.rect_listbox.bind('<<ListboxSelect>>', self.on_rect_select)
        
        # === Основное окно с изображением ===
        self.canvas_frame = ttk.Frame(self.root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg='gray20')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Привязка событий
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        
        # === Панель предпросмотра ===
        preview_frame = ttk.LabelFrame(self.root, text="Предпросмотр", padding="10")
        preview_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.preview_label = ttk.Label(preview_frame, text="Нет выделения")
        self.preview_label.pack()
        
        # === Поля для ручного ввода ===
        manual_frame = ttk.LabelFrame(self.root, text="Ручной ввод координат", padding="10")
        manual_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(manual_frame, text="X:").pack(side=tk.LEFT)
        self.entry_x = ttk.Entry(manual_frame, width=6)
        self.entry_x.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(manual_frame, text="Y:").pack(side=tk.LEFT)
        self.entry_y = ttk.Entry(manual_frame, width=6)
        self.entry_y.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(manual_frame, text="Ширина:").pack(side=tk.LEFT)
        self.entry_w = ttk.Entry(manual_frame, width=6)
        self.entry_w.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(manual_frame, text="Высота:").pack(side=tk.LEFT)
        self.entry_h = ttk.Entry(manual_frame, width=6)
        self.entry_h.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(manual_frame, text="Добавить область", 
                  command=self.add_manual_rect).pack(side=tk.LEFT, padx=10)
        
        ttk.Button(manual_frame, text="Показать", 
                  command=self.preview_manual_rect).pack(side=tk.LEFT, padx=5)
    
    def open_image(self):
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")]
        )
        
        if file_path:
            self.image = Image.open(file_path)
            self.show_image()
            self.filename = os.path.basename(file_path)
    
    def show_image(self):
        if not self.image:
            return
        
        # Масштабирование для отображения
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width < 2 or canvas_height < 2:
            canvas_width = 1000
            canvas_height = 600
        
        img_width, img_height = self.image.size
        
        # Сохраняем пропорции
        scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)
        self.scale_factor = scale
        
        if scale < 1.0:
            new_size = (int(img_width * scale), int(img_height * scale))
            display_image = self.image.resize(new_size, Image.Resampling.LANCZOS)
        else:
            display_image = self.image
            self.scale_factor = 1.0
        
        self.photo = ImageTk.PhotoImage(display_image)
        
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
        
        # Перерисовка сохранённых прямоугольников
        self.redraw_rectangles()
    
    def on_mouse_press(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        
        # Удаление предыдущего временного прямоугольника
        if self.current_rect:
            self.canvas.delete(self.current_rect)
        
        self.current_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline='red', width=2
        )
    
    def on_mouse_drag(self, event):
        if self.start_x is not None:
            cur_x = self.canvas.canvasx(event.x)
            cur_y = self.canvas.canvasy(event.y)
            
            self.canvas.coords(self.current_rect, self.start_x, self.start_y, cur_x, cur_y)
            
            # Обновление информации
            x = min(self.start_x, cur_x)
            y = min(self.start_y, cur_y)
            w = abs(cur_x - self.start_x)
            h = abs(cur_y - self.start_y)
            
            self.info_label.config(text=f"X: {int(x)}, Y: {int(y)}, W: {int(w)}, H: {int(h)}")
            
            # Предпросмотр
            self.update_preview(x, y, w, h)
    
    def on_mouse_release(self, event):
        if self.start_x is not None and self.current_rect:
            cur_x = self.canvas.canvasx(event.x)
            cur_y = self.canvas.canvasy(event.y)
            
            x = min(self.start_x, cur_x)
            y = min(self.start_y, cur_y)
            w = abs(cur_x - self.start_x)
            h = abs(cur_y - self.start_y)
            
            # Сохранение прямоугольника
            if w > 5 and h > 5:  # Минимальный размер
                self.save_rectangle(x, y, w, h)
            
            self.start_x = None
            self.start_y = None
    
    def save_rectangle(self, x, y, w, h):
        # Корректировка на масштаб
        real_x = int(x / self.scale_factor)
        real_y = int(y / self.scale_factor)
        real_w = int(w / self.scale_factor)
        real_h = int(h / self.scale_factor)
        
        rect_data = {
            'x': real_x,
            'y': real_y,
            'width': real_w,
            'height': real_h
        }
        
        self.rectangles.append(rect_data)
        self.update_rect_list()
        
        # Перерисовка
        self.redraw_rectangles()
    
    def redraw_rectangles(self):
        # Удаление всех прямоугольников кроме текущего
        self.canvas.delete("saved_rect")
        
        for i, rect in enumerate(self.rectangles):
            x = rect['x'] * self.scale_factor
            y = rect['y'] * self.scale_factor
            w = rect['width'] * self.scale_factor
            h = rect['height'] * self.scale_factor
            
            color = 'yellow' if i == self.current_index else 'green'
            
            self.canvas.create_rectangle(
                x, y, x + w, y + h,
                outline=color, width=2, tags="saved_rect"
            )
            
            # Номер
            self.canvas.create_text(
                x + 5, y + 5, text=str(i), fill=color, 
                anchor=tk.NW, tags="saved_rect"
            )
    
    def update_rect_list(self):
        self.rect_listbox.delete(0, tk.END)
        for i, rect in enumerate(self.rectangles):
            text = f"[{i}] X:{rect['x']}, Y:{rect['y']}, W:{rect['width']}, H:{rect['height']}"
            self.rect_listbox.insert(tk.END, text)
    
    def on_rect_select(self, event):
        selection = self.rect_listbox.curselection()
        if selection:
            self.current_index = selection[0]
            self.redraw_rectangles()
            
            # Показ предпросмотра
            rect = self.rectangles[self.current_index]
            x = rect['x'] * self.scale_factor
            y = rect['y'] * self.scale_factor
            w = rect['width'] * self.scale_factor
            h = rect['height'] * self.scale_factor
            self.update_preview(x, y, w, h)
    
    def update_preview(self, x, y, w, h):
        if not self.image or w < 1 or h < 1:
            return
        
        # Корректировка на масштаб
        real_x = int(x / self.scale_factor)
        real_y = int(y / self.scale_factor)
        real_w = int(w / self.scale_factor)
        real_h = int(h / self.scale_factor)
        
        # Ограничение размерами изображения
        real_x = max(0, min(real_x, self.image.width - 1))
        real_y = max(0, min(real_y, self.image.height - 1))
        real_w = max(1, min(real_w, self.image.width - real_x))
        real_h = max(1, min(real_h, self.image.height - real_y))
        
        # Вырезание области
        preview_img = self.image.crop((real_x, real_y, real_x + real_w, real_y + real_h))
        
        # Масштабирование для предпросмотра
        max_preview_size = 200
        scale = min(max_preview_size / real_w, max_preview_size / real_h, 2.0)
        if scale < 1.0:
            preview_img = preview_img.resize(
                (int(real_w * scale), int(real_h * scale)), 
                Image.Resampling.LANCZOS
            )
        
        self.preview_photo = ImageTk.PhotoImage(preview_img)
        self.preview_label.config(image=self.preview_photo, text="")
    
    def add_manual_rect(self):
        try:
            x = int(self.entry_x.get())
            y = int(self.entry_y.get())
            w = int(self.entry_w.get())
            h = int(self.entry_h.get())
            
            if w > 0 and h > 0:
                # Корректировка на масштаб отображения
                display_x = x * self.scale_factor
                display_y = y * self.scale_factor
                display_w = w * self.scale_factor
                display_h = h * self.scale_factor
                
                self.save_rectangle(display_x, display_y, display_w, display_h)
                
                # Очистка полей
                self.entry_x.delete(0, tk.END)
                self.entry_y.delete(0, tk.END)
                self.entry_w.delete(0, tk.END)
                self.entry_h.delete(0, tk.END)
            else:
                messagebox.showerror("Ошибка", "Ширина и высота должны быть > 0")
        except ValueError:
            messagebox.showerror("Ошибка", "Введите корректные числа")
    
    def preview_manual_rect(self):
        try:
            x = int(self.entry_x.get())
            y = int(self.entry_y.get())
            w = int(self.entry_w.get())
            h = int(self.entry_h.get())
            
            display_x = x * self.scale_factor
            display_y = y * self.scale_factor
            display_w = w * self.scale_factor
            display_h = h * self.scale_factor
            
            self.update_preview(display_x, display_y, display_w, display_h)
        except ValueError:
            messagebox.showerror("Ошибка", "Введите корректные числа")
    
    def save_coordinates(self):
        if not self.rectangles:
            messagebox.showwarning("Предупреждение", "Нет сохранённых областей")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("Text files", "*.txt")]
        )
        
        if file_path:
            data = {
                'image': getattr(self, 'filename', 'unknown'),
                'rectangles': self.rectangles
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            messagebox.showinfo("Успех", f"Координаты сохранены в {file_path}")
    
    def load_coordinates(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("Text files", "*.txt")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.rectangles = data.get('rectangles', [])
                self.current_index = 0
                self.update_rect_list()
                self.redraw_rectangles()
                
                messagebox.showinfo("Успех", f"Загружено {len(self.rectangles)} областей")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить файл: {e}")
    
    def clear_all(self):
        if messagebox.askyesno("Подтверждение", "Удалить все сохранённые области?"):
            self.rectangles = []
            self.current_index = 0
            self.update_rect_list()
            self.canvas.delete("saved_rect")
            self.preview_label.config(image='', text="Нет выделения")

def main():
    root = tk.Tk()
    app = SpriteSelector(root)
    root.mainloop()

if __name__ == "__main__":
    main()