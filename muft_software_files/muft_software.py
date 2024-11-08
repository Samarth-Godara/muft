#pip install openpyxl
#pip install pandas

import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd

#MUFT: Multiplier-based UnFolding Transformation of Survey Data

def multiply_dataset(df, mf=100):
  m_grps = df.groupby(['Multiplier'])
  new_df = pd.DataFrame()
  for m, grp in m_grps:
    for c in range(int(mf*m[0])):
      new_df = pd.concat([new_df, grp], ignore_index=True)
  return new_df

input_file = None
output_file = None

def select_input_file():
    global input_file
    file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
    if file_path:
        input_file = file_path
        messagebox.showinfo("Input File Selected", f"Selected input file:\n{input_file}")

def select_output_file():
    global output_file
    file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if file_path:
        output_file = file_path
        messagebox.showinfo("Output File Selected", f"Selected output file:\n{output_file}")

def start_transformation():
    if not input_file:
        messagebox.showwarning("No Input File", "Please select an input file first.")
        return
    if not output_file:
        messagebox.showwarning("No Output File", "Please select an output file location.")
        return

    try:
        # Load input data
        df = pd.read_excel(input_file)
        
        mf = int(mf_entry.get())
        
        muft_df = multiply_dataset(df.copy(), mf)

        # Transform data here (this example just saves it as .csv)
        muft_df.to_csv(output_file, index=False)
        
        messagebox.showinfo("Transformation Complete", f"Data has been saved to:\n{output_file}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

root = tk.Tk()
root.title("MUFT Transformation")

title_label = tk.Label(root, text="Welcome to",font=("Helvetica", 10, "normal"))
title_label.grid(row=0, column=0, padx=10, pady=10,columnspan=2)

title_label = tk.Label(root, text="MUFT",font=("Helvetica", 34, "bold"))
title_label.grid(row=1, column=0, padx=10, pady=0,columnspan=2)

title_label = tk.Label(root, text="Multiplier-based UnFolding \nTransformation of Survey Data",font=("Helvetica", 14, "bold"))
title_label.grid(row=2, column=0, padx=10, pady=10,columnspan=2)

select_input_button = tk.Button(root, text="Select Input\n(.xlsx File)", command=select_input_file,font=("Helvetica", 12, "bold"), borderwidth=5)
select_input_button.grid(row=3, column=0, padx=20, pady=10)

select_output_button = tk.Button(root, text="Select Output\n(File Location)", command=select_output_file,font=("Helvetica", 12, "bold"), borderwidth=5)
select_output_button.grid(row=3, column=1, padx=20, pady=10)

mf_label = tk.Label(root, text="Multiplication Factor: ",font=("Helvetica", 10, "bold"))
mf_label.grid(row=5, column=0,columnspan=2, padx=10, pady=10)
mf_entry = tk.Entry(root, width=5)
mf_entry.insert(0, "100")
mf_entry.grid(row=5, column=1,columnspan=1, padx=50, pady=10, sticky='e')

transform_button = tk.Button(root, text="Start Transformation", command=start_transformation,font=("Helvetica", 14, "bold"), borderwidth=10)
transform_button.grid(row=6, column=0, padx=10, pady=10,columnspan=2)

credits_label = tk.Label(root, text="Developed by: ICAR-IASRI, New Delhi, India",font=("Helvetica", 8, "normal"))
credits_label.grid(row=7, column=0, padx=10, pady=10,columnspan=2)


root.mainloop()
