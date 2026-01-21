# -*- coding: utf-8 -*-
#Code written by: Yingtong Lin

##General overview
##Pred_N is an interactive desktop application designed to predict the adsorption capacity of gas molecules in metal-organic frameworks (MOFs) or other porous crystalline materials. Its core computation is based on a trained LightGBM (LGBM) model. Below is the source code for the interface design. 

## Import the required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tkinter as tk
import ttkbootstrap as ttk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import webbrowser
import joblib
import sys
import os



# ================= 配置参数 ================

## Load the trained XGB model
model = joblib.load("../model/xgb.pt") ## The detailed code for model training is in “XGB_code.py”

## The layout design of the main interface
root= tk.Tk()
root.title("Predict N of material on XGB")
root.resizable(True,True) ## The window size can be changed
canvas1 = tk.Canvas(root, width = 800, height =600) ## Main window size
canvas1.pack()

## The plate layout within the main interface
# 主要框架（粗黑框）
re1 = canvas1.create_rectangle(50, 20, 750, 550, outline='black', width=1.5)
re2 = canvas1.create_rectangle(50.25, 450, 750, 550, outline='black', width=2.3)

# 辅助框架（darkgray）
re3 = canvas1.create_rectangle(70, 50, 730, 430, outline='darkgray', width=1)
re4 = canvas1.create_rectangle(400, 330, 730, 430, outline='darkgray', width=1)
re5 = canvas1.create_rectangle(50, 450, 750, 500, outline='darkgray', width=1)
re6 = canvas1.create_rectangle(50, 500, 750, 550, outline='darkgray', width=1)

# 标签
label_B = tk.Label(root, font=('Times New Roman', 12), text='Predicted results')
canvas1.create_window(470, 330, window=label_B)

label_B = tk.Label(root, font=('Times New Roman', 12), text='Predicted results')
canvas1.create_window(650, 500, window=label_B)

label_B = tk.Label(root, font=('Times New Roman', 10), text='Author：Yingtong Lin,Zhiwei Qiao,Guangzhou University')
canvas1.create_window(600, 580, window=label_B)


## Message box (Related literature on molecular physical properties)
def cmx1():
    window = tk.Tk()   #创建一个新的Tkinter窗口
    window.title('Warm prompt')  #设置窗口标题   
    window.geometry('350x250')  #设置窗口大小
    
    #创建一个标签，显示文本和超链接
    link = tk.Label(window, text='The physical property of material \nare known from the literature:\nhttps://doi.org/10.1016/j.memsci.2024.123612'
                    , font=('Times New Roman',10),anchor="center")
    link.place(x=30, y=50) #将标签放置在窗口的位置
    
    #定义一个函数，用于在点击超链接时打开浏览器访问链接
    def open_url(event):
        webbrowser.open("https://doi.org/10.1016/j.memsci.2024.123612", new=0)         
    link.bind("<Button-1>", open_url)    #将点击事件绑定在标签上
    
#创建按钮并放置在主界面的画布上
btn1 = tk.Button(root, text='Tooltip', font=('Times New Roman', 10), command=cmx1)
canvas1.create_window(100, 410, window=btn1)

##Message box (Instructions for Prediction of a single material adsorption capacity)
def resize(w, h, w_box, h_box, pil_image): 
  f1 = 1*w_box/w 
  f2 =1*h_box/h  
  factor = min([f1, f2])  
  width = int(w*factor)  
  height = int(h*factor)  
  return pil_image.resize((width, height),Image.LANCZOS)
       
w_box = 600  
h_box = 450    


# 加载并缩放图片
photo1 = Image.open("../Img/full_name.png")
w, h = photo1.size
photo1_resized = resize(w, h, w_box, h_box, photo1)    # 假设 resize 是自定义函数
tk_image1 = ImageTk.PhotoImage(photo1_resized)


def cmx2():
    top2=tk.Toplevel() 
    top2.title('Instructions for use') 
    top2.geometry('620x500') 
    lab_1 = ttk.Label(top2,image=tk_image1) 
    lab_1.place(x=15, y=8) 
    top2.mainloop()  
  
btn2=tk.Button(root, text='READ ME',font=('Times New Roman',10), command=cmx2)
canvas1.create_window(90, 36, window=btn2)

## Message box (Instructions for batch Prediction of material adsorption capacity)
def resize(w, h, w_box, h_box, pil_image):
  f1 = 1*w_box/w 
  f2 =1*h_box/h  
  factor = min([f1, f2])  
  width = int(w*factor)  
  height = int(h*factor)  
  return pil_image.resize((width, height),Image.LANCZOS)      
w_box = 600  
h_box = 500    

# 加载并缩放图片
photo2 = Image.open("../Img/sample_file.png")
w, h = photo2.size
photo2_resized = resize(w, h, w_box, h_box, photo2)
tk_image2 = ImageTk.PhotoImage(photo2_resized)



def cmx3():
    top1 = tk.Toplevel()
    top1.title('Instructions for use')     
    top1.geometry('680x580')
    
    # 创建一个框架用于居中内容
    frame = tk.Frame(top1)
    frame.pack(expand=True, fill='both')
    
    # 创建文字标签
    lab2 = tk.Label(frame, text=' Create the data you want to compute in the format given below. \n An example is the prediction of N_NH3', font=('Times New Roman', 15), justify='left')
    lab2.pack(pady=10, anchor='w')  # 使用 pack 布局管理器，并添加上下边距
    
    # 创建图片标签
    lab3 = ttk.Label(frame, text="photo:", image=tk_image2)
    lab3.pack(pady=10, anchor='w')  # 使用 pack 布局管理器，并添加上下边距
    
    # 创建另一个文字标签
    lab4 = tk.Label(frame, text=' After generating the file, you may click the import file button on the interface.\n The forecasted data will be kept in "Result/Batch_Predicted_N.xlsx".', font=('Times New Roman', 15), justify='left')
    lab4.pack(pady=10, anchor='w')  # 使用 pack 布局管理器，并添加上下边距
    
    top1.mainloop() 

btn3 = tk.Button(root, text='READ ME', font=('Times New Roman', 10), command=cmx3)
canvas1.create_window(100, 475, window=btn3)



## Sets the label and entry for entering the nine descriptor 
label_Z = tk.Label(root,font=('Times New Roman',13),text='Predict adsorption capacity of material')
canvas1.create_window(400, 20, window=label_Z)

label_L = tk.Label(root,font=('Times New Roman',11),text='Physical property of material')
canvas1.create_window(400, 50, window=label_L)

label1 = tk.Label(root,font=('Times New Roman',10 ,"italic"),text='φ：') ## create 1st label box 
canvas1.create_window(160, 90, window=label1)
entry1 = tk.Entry (root,font=('Times New Roman',10),width=12,justify='center') ## create 1st entry box 
canvas1.create_window(245, 90, window=entry1)

label2 = tk.Label(root,font=('Times New Roman',10), text='VSA (m^2/cm^3): ') ## create 2st label box 
canvas1.create_window(510, 90, window=label2)
entry2 = tk.Entry (root,font=('Times New Roman',10),width=12,justify='center') ## create 2nd entry box
canvas1.create_window(598, 90, window=entry2)

label3 = tk.Label(root,font=('Times New Roman',10), text='LCD (Å): ') ## create 3st label box 
canvas1.create_window(160, 185, window=label3)
entry3 = tk.Entry (root,font=('Times New Roman',10),width=12,justify='center') ## create 3nd entry box
canvas1.create_window(245, 185, window=entry3)

label4 = tk.Label(root, font=('Times New Roman',10),text='PLD (Å): ') ## create 4st label box 
canvas1.create_window(505, 185, window=label4) 
entry4 = tk.Entry (root,font=('Times New Roman',10),width=12,justify='center') ## create 4nd entry box
canvas1.create_window(598, 185, window=entry4)

label5 = tk.Label(root,font=('Times New Roman',10,"italic"), text='ρ') ## create 5st label box 
canvas1.create_window(130, 280, window=label5)
l0 = tk.Label(root,font=('Times New Roman',10), text='(kg/cm^3): ') 
canvas1.create_window(170, 280, window=l0)
entry5 = tk.Entry (root,font=('Times New Roman',10),width=12,justify='center') ## create 5nd entry box
canvas1.create_window(245, 280, window=entry5)

label6 = tk.Label(root, font=('Times New Roman',10,"italic"),text='K') ## create 1st 6abel box 
canvas1.create_window(470,280, window=label6)
l1 = tk.Label(root, font=('Times New Roman',10),text='(mol/(kg•Pa)):')
canvas1.create_window(520,280, window=l1) 
entry6 = tk.Entry (root,font=('Times New Roman',10),width=12,justify='center') ## create 6nd entry box
canvas1.create_window(598,280, window=entry6)

label7 = tk.Label(root, font=('Times New Roman', 10, "italic"), text='Qst')  
canvas1.create_window(150, 380, window=label7) 
l2 = tk.Label(root, font=('Times New Roman', 10), text='(kJ/mol): ')
canvas1.create_window(240, 380, window=l2) 
entry7 = tk.Entry(root, font=('Times New Roman', 10), width=12, justify='center')  
canvas1.create_window(245, 380, window=entry7)  


## Main interface for input of nine descriptor values (a single molecule diffusivity)
def values():       
    global New_φ #our 1st input variable    
    New_φ = float(entry1.get()) 
    
    global New_VSA #our 2nd input variable
    New_VSA = float(entry2.get()) 
    
    global New_PLD #our 2nd input variable
    New_PLD = float(entry3.get()) 
    
    global New_LCD #our 2nd input variable
    New_LCD = float(entry4.get()) 
    
    global New_ρ #our 2nd input variable
    New_ρ =float(entry5.get()) 
    
    global New_K  # our 2nd input variable
    New_K = float(entry6.get())
    if New_K <= 0:  # 防御非正值
        messagebox.showerror("Error", "K值必须大于0！")
        return
    New_K = np.log10(New_K)  # 取以10为底的对数
    
    global New_Qst #our 2nd input variable
    New_Qst = float(entry7.get()) 


## LGBM Algorithm (The predictions of a single molecule diffusivity)   
    Pred_N = model.predict([[New_φ, New_VSA, New_PLD, New_LCD, New_ρ, New_K,
                          New_Qst]])      
   
    Pred_N = Pred_N[0]
 # 保留两位小数（四舍五入）
    Pred_N_rounded = round(Pred_N, 2)

# 格式化输出字符串
    Prediction_result = f"{Pred_N_rounded:.2f}"
    

    label_Prediction = tk.Label(root, font=('Times New Roman',12),width=15,height=2,
                                text= Prediction_result)
    canvas1.create_window(580, 380, window=label_Prediction)


    ## N label
    lbo1 = tk.Label(root, font=('Times New Roman', 12, "italic"), text='N:')
    canvas1.create_window(520, 380, window=lbo1)  # 调整标签位置，避免挡住按钮
    
    ## unit label
    lbo2 = tk.Label(root, font=('Times New Roman', 12),
                    text='(mol/kg)')
    canvas1.create_window(680, 380, window=lbo2)  # 调整标签位置，避免挡住按钮

## button to call the 'values' command above       
button1 = tk.Button (root,font=('Times New Roman',10), text='Predicted N',command=values) 
canvas1.create_window(450, 380, window=button1)


## Batch prediction of material adsorption capacity
label_Z1 = tk.Label(root,font=('Times New Roman',12),text='Batch prediction of material adsorption capacity')
canvas1.create_window(400, 450, window=label_Z1)

## Open File
def open_file():
    filename = filedialog.askopenfilename(title='open exce')
    entry_filename.delete(0,"end")
    entry_filename.insert('insert', filename)
 
button_import = tk.Button(root, text="Import File",font=('Times New Roman',10),command=open_file)
canvas1.create_window(180, 475, window=button_import)
 
## Import File
entry_filename = tk.Entry(root,font=('Times New Roman',10),width=30)
canvas1.create_window(520, 480, window=entry_filename)

def print_file():
    a = entry_filename.get()
    
    # 检查文件路径有效性
    if not os.path.exists(a):
        messagebox.showerror("Error", "文件路径不存在！")
        return
    
    # 加载数据
    try:
        pred_data1 = pd.read_excel(a)
    except Exception as e:
        messagebox.showerror("Error", f"文件读取失败：{str(e)}")
        return
    
    # 检查数据是否为空
    if pred_data1.empty:
        messagebox.showerror("Error", "文件内容为空！")
        return
    
    # 删除缺失值
    pred_data = pred_data1.dropna(axis=0)
    if pred_data.empty:
        messagebox.showerror("Error", "数据经过缺失值处理后为空！")
        return
    
    if (pred_data['K'] <= 0).any():
       messagebox.showerror("Error", "K列存在非正值，请检查数据！")
       return
    
    # 检查列名是否匹配
    required_columns = ['φ', 'VSA', 'LCD', 'PLD', 'ρ', 'K', 'Qst']
    missing_columns = [col for col in required_columns if col not in pred_data.columns]
    if missing_columns:
        messagebox.showerror("Error", f"缺失列：{', '.join(missing_columns)}")
        return
    
    # 构建特征矩阵
    df = pd.DataFrame(pred_data, columns=required_columns)
    X_pred = df[['φ', 'VSA', 'LCD', 'PLD', 'ρ', 'K', 'Qst']].astype(float)
    X_pred['K'] = np.log10(X_pred['K'])  # 关键修改点
    X_pred = X_pred.astype(float)
   
    # 检查特征矩阵是否为空
    if X_pred.shape[0] == 0:
        messagebox.showerror("Error", "特征矩阵无有效样本！")
        return
    
    # 标准化
    try:
        transfer = StandardScaler()
        X_pred = transfer.fit_transform(X_pred)
    except Exception as e:
        messagebox.showerror("Error", f"标准化失败：{str(e)}")
        return
    
    # 模型预测
    try:
        Y_predict2 = model.predict(X_pred)
    except Exception as e:
        messagebox.showerror("Error", f"模型预测失败：{str(e)}")
        return
    
    # 结果保存

    d1 = pd.DataFrame({'Pred_N': Y_predict2})
    newdata = pd.concat([pred_data, d1], axis=1)
    
    # 确保输出目录存在
    output_dir = os.path.dirname("../Result/Batch_Predicted_N.xlsx")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        newdata.to_excel("../Result/Batch_Predicted_N.xlsx", index=False)
    except Exception as e:
        messagebox.showerror("Error", f"结果保存失败：{str(e)}")
        return
    
    # 显示成功信息
    label_P = tk.Label(root, font=('Times New Roman', 12),
                       text='Predicted results have default stored in:\nResult/Batch_Predicted_N.xlsx', bg='blue')
    canvas1.create_window(450, 525, window=label_P)
    
## Prediction button
but_pre=tk.Button(root,font=('Times New Roman',10)
             , text='Batch Predicted N', command=print_file)
canvas1.create_window(120, 525, window=but_pre)

root.mainloop() 
