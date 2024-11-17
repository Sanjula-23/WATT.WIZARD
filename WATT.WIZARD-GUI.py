# This Code is for Energy monitoring System with the GUI. 
# Dinalofcl - 2024

import customtkinter as ctk
from tkinter import messagebox, filedialog
import pandas as pd 
import numpy as np
import os, time
from datetime import datetime
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from PIL import Image, ImageTk


class EnergyMonitorGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("WATT.WIZARD - Energy Monitoring System")
        self.geometry("1600x950")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("/Users/sanjulaweerasekara/Desktop/WATT.WIZARD Project/dark-blue.json")

        # Font
        header_font = ctk.CTkFont(family="Apple SD Gothic Neo", size=26, weight="bold")


        # GIF (You an add your own GIF here)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        gif_path = os.path.join(script_dir, "WATT.WIZARD1.gif")

        self.gif_image = Image.open(gif_path)
        self.after(100, self.start_animation)    
                     
        self.background_label = ctk.CTkLabel(self)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)

        # Side bar 
        self.sidebar_frame = ctk.CTkFrame(self, width=300, corner_radius=15)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, pady=(80, 10), padx=(20, 10), sticky="nsew")

        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        # Icon 
        icon_path = os.path.join(script_dir, "verify.png")
        
        icon_image = Image.open(icon_path)
        icon_image = icon_image.resize((24,24))

        self.icon_photo = ctk.CTkImage(light_image=icon_image, dark_image=icon_image, size=(24, 24))

        self.clock_label = ctk.CTkLabel(self.sidebar_frame, font=ctk.CTkFont(size=20, weight="bold"))
        self.clock_label.grid(row=4, column=0, padx=20, pady=(10, 20))  # Position it below the appearance mode menu
        
        # Start the clock update loop
        self.update_clock()

        # LOGO with Icon
        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame,
            text=" WATT.WIZARD ",
            font=ctk.CTkFont(size=20, weight="bold"),
            image=self.icon_photo,
            compound="right"
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Side bar buttons
        self.load_button = ctk.CTkButton(self.sidebar_frame, text=" Load Data", command=self.load_data)
        self.load_button.grid(row=1, column=0, padx=20, pady=10)

        self.analyze_button = ctk.CTkButton(self.sidebar_frame, text="Analyze Data", command=self.analyze_data)
        self.analyze_button.grid(row=2, column=0, padx=20, pady=10)

        self.predict_button = ctk.CTkButton(self.sidebar_frame, text="Predict Future", command=self.predict_future)
        self.predict_button.grid(row=3, column=0, padx=20, pady=10)


        # Appearance modes
        self.appearance_mode_label = ctk.CTkLabel(self.sidebar_frame, text="Themes 🔆", anchor="w")
        self.appearance_mode_label.grid(row =5 , column=0, padx=20, pady=(10, 0))
        self.appearance_mode_menu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Dark", "Light", "System"], command=self.change_appearance_mode)
        self.appearance_mode_menu.grid(row=6, column=0, padx=20, pady=(10,10))

        # Main content 
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, padx=100, pady=30, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)

        # header label
        self.header_label = ctk.CTkLabel(self.main_frame, text=" Energy Consumption Monitoring Dashboard ", font=header_font)
        self.header_label.grid(row=0, column=0, padx=20, pady=20)

        # tab view
        self.tabview = ctk.CTkTabview(self.main_frame)
        self.tabview.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")

        # Creat tab
        self.tab_data = self.tabview.add(" Data View ")
        self.tab_analysis = self.tabview.add(" Analysis ")
        self.tab_prediction = self.tabview.add(" Prediction ")

        # Setup tab contents
        self._setup_data_tab()
        self._setup_analysis_tab()
        self._setup_prediction_tab()

        # bring frames to the front
        self.sidebar_frame.lift()
        self.main_frame.lift()

    # Animation 
    def start_animation(self):
        if self.gif_image:
            self.animate_gif()
            
    def animate_gif(self, counter=0):
        try:
            self.gif_image.seek(counter)
            frame_image = self.gif_image.copy().resize((self.winfo_width(), self.winfo_height()), Image.LANCZOS)


            frame = ImageTk.PhotoImage(frame_image)

            self.background_label.configure(image=frame)
            self.background_label.image = frame

            counter += 1

        except EOFError:
            counter = 0

        
        self.gif_image.seek(0)
        self.after(200, self.animate_gif, counter)
    
    def _setup_data_tab(self):
        self.data_controls_frame = ctk.CTkFrame(self.tab_data)
        self.data_controls_frame.pack(padx=20, pady=(50,10), fill="x")

        # search
        #self.search_label = ctk.CTkFrame(self.tab_data)
        #self.search_label.pack(side="left", padx=7)

        self.search_label = ctk.CTkLabel(self.data_controls_frame, text="Search :")
        self.search_label.pack(side="left", padx=(10,5))

        self.search_entry = ctk.CTkEntry(self.data_controls_frame, width=400)               
        self.search_entry.pack(side="left", padx=10)

        self.search_button = ctk.CTkButton(self.data_controls_frame, text="Search ", command=self.search_data)
        self.search_button.pack(side="left", padx=5)

        self.clear_search_button = ctk.CTkButton(self.data_controls_frame, text="Clear Search", command=self.clear_search)
        self.clear_search_button.pack(side="left", padx=5)

        # scroll bar
        self.data_text = ctk.CTkTextbox(self.tab_data, width=800, height=400)
        self.data_text.pack(padx=20, pady=(10, 20), fill="both", expand=True)

    # Search Data Tab
    def search_data(self):
        search_query = self.search_entry.get()
        if hasattr(self, 'data') and self.data is not None and search_query:
            try:
                # Attempt to parse search_query as a date
                search_date = pd.to_datetime(search_query, errors='coerce')

                # Filter data based on search query
                if not pd.isnull(search_date):
                # Search for rows where the index (date) matches the search date
                    filtered_data = self.data[self.data.index == search_date]
                else:
                # Search for the string in the data as usual
                    filtered_data = self.data[self.data.apply(
                    lambda row: row.astype(str).str.contains(search_query, case=False).any(), axis=1)]
            
            # Display filtered data
                self.display_data(filtered_data)
            except Exception as e:
                messagebox.showerror("Error", f"Error in search: {e}")
        else:
            messagebox.showinfo("Info", "Please enter a search query and ensure data is loaded.")


    def clear_search(self):
        if hasattr(self, 'data'):
            self.display_data(self.data)


    def _setup_analysis_tab(self):
        # Analysis results
        self.analysis_frame = ctk.CTkFrame(self.tab_analysis)
        self.analysis_frame.pack(padx=20, pady=20, fill="both", expand=True)

        self.analysis_text = ctk.CTkTextbox(self.analysis_frame, width=800, height=400)
        self.analysis_text.pack(padx=20, pady=20, fill="both", expand=True)


    def _setup_prediction_tab(self):
        # Prediction inputs and results
        self.prediction_frame = ctk.CTkFrame(self.tab_prediction)
        self.prediction_frame.pack(padx=20, pady=20, fill="both", expand=True)

        # Hours input
        self.hours_frame = ctk.CTkFrame(self.prediction_frame)
        self.hours_frame.pack(padx=20, pady=10)

        self.hours_label = ctk.CTkLabel(self.hours_frame, text="Prediction Hours:")
        self.hours_label.pack(side="left", padx=5)

        self.hours_entry = ctk.CTkEntry(self.hours_frame)
        self.hours_entry.pack(side="left", padx=5)
        self.hours_entry.insert(0, "24")

        # Results display
        self.prediction_text = ctk.CTkTextbox(self.prediction_frame, width=800, height=400)
        self.prediction_text.pack(padx=20, pady=20, fill="both", expand=True) 

    # Live clock
    def update_clock(self):
        # Get the current time
        current_time = datetime.now().strftime("%Y/%m/%d \n%H:%M:%S")
        
        # Update the clock label
        self.clock_label.configure(text=current_time)
        
        # Schedule the function to update every 1000 ms (1 second)
        self.after(1000, self.update_clock)


    def display_data(self, data):
        self.data_text.delete("1.0", "end")

        # Create a formatted string with aligned columns
        # Get the string representation of the DataFrame with all rows
        df_string = data.to_string()

        # Add a header with data info
        header = f"Total Rows: {len(data)}\nTotal Columns: {len(data.columns)}\n\n"

        # Display the data with header
        self.data_text.insert("1.0", header + df_string)

        # Scroll to the top
        self.data_text.yview_moveto(0)


    def change_appearance_mode(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode.lower())

    
    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                self.data = pd.read_csv(file_path, parse_dates=['timestamps'])
                self.data = self.data.set_index('timestamps')
                self.data = self.clean_data(self.data)

                # Display all rows in data tab
                self.display_data(self.data)

                messagebox.showinfo("Success", f"Data loaded and cleaned successfully!\nTotal rows: {len(self.data)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error loading data: {str(e)}")

    
    def clean_data(self, data):
        if 'consumption' in data.columns:
            data['consumption'] = data['consumption'].clip(lower=0)
            mean_value = data['consumption'].mean()

            data['consumption'] = data['consumption'].fillna(mean_value)
            data.loc[data['consumption'] > 10000, 'consumption'] = mean_value

            # inter quartile method 
            Q1 = data['consumption'].quantile(0.25)
            Q3 = data['consumption'].quantile(0.75)
            IQR = Q3 - Q1
            upper_threshold = Q3 + 1.5 * IQR

            data['consumption'] = np.where(data['consumption'] > upper_threshold, mean_value, data['consumption'])
            
        return data
    

    def analyze_data(self):
        if self.data is None:
            messagebox.showwarning("Please upload the data first!")
            return
        
        hourly_consumption = self.data['consumption'].resample('h').sum()

        analysis_text = "Analysis Results:\n\n"
        analysis_text += f"Highest Usage: {hourly_consumption.max():.2f} kWh on {hourly_consumption.idxmax()}\n"
        analysis_text += f"Lowest Usage: {hourly_consumption.min():.2f} kWh on {hourly_consumption.idxmin()}\n"
        analysis_text += f"Total Consumption: {hourly_consumption.sum():.2f} kWh\n"
        analysis_text += f"Maximum Day Consumption %: {(hourly_consumption.max() / hourly_consumption.sum() * 100):.2f}%\n"

        # Create separate window for the graph
        graph_window = ctk.CTkToplevel(self)
        graph_window.title("Energy Consumption Analysis")
        graph_window.geometry("1280x720")  

        # Create figure with larger size
        fig = plt.figure(figsize=(12, 8), dpi=100)
        ax = fig.add_subplot(111)

        # Plot with more detailed formatting
        hourly_consumption.plot(ax=ax, linewidth=2)
        ax.set_title("Hourly Energy Consumption", fontsize=14, pad=20)
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Consumption (kWh)", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='both', which='major', labelsize=10)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)

        # Add padding to prevent label cutoff
        plt.tight_layout()

        # create tool bar frame
        toolbar_frame = ctk.CTkFrame(graph_window)
        toolbar_frame.pack(side="top", fill="x")

        canvas = FigureCanvasTkAgg(fig, master=graph_window)
        canvas.draw()

        # Add navigation bar
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()

        # Pack the canvas
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)


    # Predict the Future here. ML Model is here.
    def predict_future(self):
        if self.data is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return

        try:
            hours = int(self.hours_entry.get())
            if hours <= 0:
                raise ValueError("Hours must be positive")

            # Prepare features
            self.data['hour'] = self.data.index.hour
            self.data['days_of_week'] = self.data.index.dayofweek
            self.data['month'] = self.data.index.month
            self.data['week_of_year'] = self.data.index.isocalendar().week

            features = ['hour', 'days_of_week', 'month', 'week_of_year']
            X = self.data[features]
            y = self.data['consumption']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

            # Train model
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100,max_features= None,max_depth=10,min_samples_split=2, min_samples_leaf=1,bootstrap=True, random_state=100)  # Pre trained Values
            model.fit(X_train , y_train)

            # Generate future timestamps
            last_timestamp = self.data.index[-1]
            future_data = []

            for i in range(1, hours + 1):
                next_hour = last_timestamp + pd.Timedelta(hours=i)
                future_data.append({
                    'hour': next_hour.hour,
                    'days_of_week': next_hour.dayofweek,
                    'month': next_hour.month,
                    'week_of_year': next_hour.isocalendar().week
                })

            future_df = pd.DataFrame(future_data)
            predictions = model.predict(future_df)

            # Display results
            self.prediction_text.delete("1.0", "end")
            self.prediction_text.insert("1.0", "Predicted Consumption:\n\n")

            for i, pred in enumerate(predictions):
                timestamp = last_timestamp + pd.Timedelta(hours=i+1)
                self.prediction_text.insert("end", f"{timestamp}: {pred:.2f} kWh\n")

            # Create separate window for the prediction graph
            pred_graph_window = ctk.CTkToplevel(self)
            pred_graph_window.title("Energy Consumption Forecast")
            pred_graph_window.geometry("1280x720")  

            # Create figure with larger size
            fig = plt.figure(figsize=(12, 8), dpi=100)
            ax = fig.add_subplot(111)

            # Plot with more detailed formatting
            ax.plot(
                self.data.index,
                self.data['consumption'],
                label='Historical',
                linewidth=2
            )

            future_timestamps = [
                last_timestamp + pd.Timedelta(hours=i+1)
                for i in range(hours)
            ]
            ax.plot(
                future_timestamps,
                predictions,
                'r--',
                label='Predicted',
                linewidth=2
            )

            ax.set_title("Energy Consumption Forecast",weight="bold", fontsize=16, pad=20)
            ax.set_xlabel("Time", fontsize=12)
            ax.set_ylabel("Consumption (kWh)", fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, linestyle='-', alpha=0.7)
            ax.tick_params(axis='both', which='major', labelsize=10)

            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)

            # Add padding to prevent label cutoff
            plt.tight_layout()

            # Create toolbar frame
            toolbar_frame = ctk.CTkFrame(pred_graph_window)
            toolbar_frame.pack(side="top", fill="x")

            # Create canvas with scrollable region
            canvas = FigureCanvasTkAgg(fig, master=pred_graph_window)
            canvas.draw()

            # Add navigation toolbar
            from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()

            # Pack the canvas
            canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction error: {str(e)}")


if __name__ == "__main__":
    app = EnergyMonitorGUI()
    app.mainloop()




