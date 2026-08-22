from tkinter import *

window = Tk()
window.title('TEXT BOX')
window.geometry("400x400")

#Clear function
def clear():
    text_box.delete(1.0 , END) #Text boxes start at 1.0

text_box = Text(window,width=30 , height=10)
text_box.pack(pady=20)

button_frame = Frame(window)
button_frame.pack()

clear_button =Button(button_frame, text="Clear Text" , command=clear)
clear_button.grid(row=0, column =1)

#Keep the application alive and responsive
window.mainloop()