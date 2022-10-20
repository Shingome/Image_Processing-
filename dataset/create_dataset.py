import tkinter as tk
from random import randint as rd
import numpy as np
from PIL import ImageTk, Image


class App:
    def save(function):
        def saved(self):
            if self.dataset:
                dataset = np.asarray(self.dataset, dtype=object)
                try:
                    old_dataset = np.load('/training/dataset.npy',
                                          allow_pickle=True)
                    dataset = np.vstack((dataset, old_dataset))
                    np.save('datasets\\', dataset)
                except:
                    np.save('datasets\\', dataset)
                    print('Создан новый файл')
                print(dataset.shape[0])
            function(self)

        return saved

    def __init__(self):
        self.window = tk.Tk()
        self.window.resizable(False, False)
        self.window.title('dataset')
        self.window.geometry('550x320')

        self.dataset = []

        button_135 = tk.Button(self.window, text='135', width=10, height=5, command=lambda: self.save_image(4))
        button_135.place(x=450, y=10)

        button_45 = tk.Button(self.window, text='45', width=10, height=5, command=lambda: self.save_image(3))
        button_45.place(x=450, y=110)

        button_180 = tk.Button(self.window, text='180', width=10, height=5, command=lambda: self.save_image(2))
        button_180.place(x=350, y=10)

        button_90 = tk.Button(self.window, text='90', width=10, height=5, command=lambda: self.save_image(1))
        button_90.place(x=350, y=110)

        button_none = tk.Button(self.window, text='none', width=10, height=5, command=lambda: self.save_image(0))
        button_none.place(x=350, y=210)

        button_next = tk.Button(self.window, text='next', width=10, height=5, command=lambda: self.next_image())
        button_next.place(x=450, y=210)

        self.images = np.load('images_for_dataset.npy')

        self.canvas = tk.Canvas(self.window, height=300, width=300)
        self.image_prev = self.images[rd(0, np.shape(self.images)[0])]
        self.image = Image.fromarray(self.image_prev)
        self.image = self.image.resize((300, 300), resample=Image.NEAREST)
        self.photo = ImageTk.PhotoImage(self.image)
        self.image = self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        self.canvas.place(x=10, y=10)

        self.window.mainloop()

    @save
    def next_image(self):
        self.dataset = []
        self.image_prev = self.images[rd(0, np.shape(self.images)[0])]
        self.image = Image.fromarray(self.image_prev)
        self.image = self.image.resize((300, 300), resample=Image.NEAREST)
        self.photo = ImageTk.PhotoImage(self.image)
        self.image = self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        self.canvas.place(x=10, y=10)

    def save_image(self, answer):
        image_mat = np.asarray(self.image_prev) / 255
        image_array = np.reshape(image_mat, (1, 64))
        self.dataset.append([answer, image_array])
        self.dataset.append([answer, np.ones((1, 64)) - image_array])

        if answer in (3, 4):
            image_array = image_mat.T
            image_array = np.reshape(image_array, (1, 64))
            self.dataset.append([answer, image_array])
            self.dataset.append([answer, np.ones((1, 64)) - image_array])

            image_array = np.rot90(np.rot90(image_mat))
            image_array = np.reshape(image_array, (1, 64))
            self.dataset.append([answer, image_array])
            self.dataset.append([answer, np.ones((1, 64)) - image_array])

            image_array = image_mat.T
            image_array = np.reshape(image_array, (1, 64))
            self.dataset.append([answer, image_array])
            self.dataset.append([answer, np.ones((1, 64)) - image_array])

        else:
            image_array = np.flipud(image_mat)
            image_array = np.reshape(image_array, (1, 64))
            self.dataset.append([answer, image_array])
            self.dataset.append([answer, np.ones((1, 64)) - image_array])

            image_array = np.fliplr(image_mat)
            image_array = np.reshape(image_array, (1, 64))
            self.dataset.append([answer, image_array])
            self.dataset.append([answer, np.ones((1, 64)) - image_array])

            image_array = np.flipud(image_mat)
            image_array = np.reshape(image_array, (1, 64))
            self.dataset.append([answer, image_array])
            self.dataset.append([answer, np.ones((1, 64)) - image_array])

        self.next_image()


app = App()
