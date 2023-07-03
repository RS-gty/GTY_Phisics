import os
from tkinter import *
from tkinter import messagebox

from Simulate import *


root = Tk()
root.title("tkinter + Matplotlib")
root.geometry('700x750')


def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        root.destroy()


def Func1():
    os.system('py Simulate.py')


Label(root, text='tkinter & Matplotlib动态示例').place(x=0, y=0, width=700, height=50)
Bu1 = Button(root, command=Func1)
Bu1.pack()

root.protocol('WM_DELETE_WINDOW', on_closing)
root.mainloop()
