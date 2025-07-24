import tkinter as tk
from tkinter import scrolledtext
from tkinter import messagebox
import yaml
import os
from ollama import chat, Client
import re
import ast
import time
import openai
from openai import OpenAI

import yamlloader

drives = []

chosen_conversation_file = None
current_dir = os.path.dirname(os.path.abspath(__file__))

def init_conversation_file():
    """
    Description:
        Initializes the conversation file by finding the first available filename.
        Should be called once at the start of the program.
    Inputs:
        None
    Outputs:
        Sets the global variable 'chosen_conversation_file' with the file path.
    """
    global chosen_conversation_file
    base_dir = current_dir
    base_name = "conversation_"
    extension = ".yaml"

    i = 1
    while True:
        file_name = f"{base_name}{i}{extension}"
        file_path = os.path.join(base_dir, file_name)
        if not os.path.exists(file_path):
            chosen_conversation_file = file_path
            break
        i += 1

def save_conversation(conversation):
    """
    Description:
        Saves the conversation to the file chosen at initialization.
        Can be called multiple times per run.
    Inputs:
        conversation (list): List of conversation messages.
    Outputs:
        Writes the conversation to the YAML file.
    """
    if chosen_conversation_file is None:
        raise RuntimeError("You must call init_conversation_file() before save_conversation()")

    with open(chosen_conversation_file, "w", encoding="utf-8") as file:
        yaml.dump(conversation, file, allow_unicode=True)

class LLMmodel():
    """
    Description:
        Creates LLM model instance and manages communication.
    Parameters:
        model (str): Model name.
        initial_prompt (str): Initial prompt for the LLM.
    """
    def __init__(self, model, initial_prompt):
        self.model = model
        self.initial_prompt = initial_prompt
        if model == "phi4:14b" or model == "Qwen3:30b":
            self.client = Client(host="http://HOST")
        else: 
            self.client = OpenAI(base_url="https://openrouter.ai/api/v1",
                                 api_key="API_KEY",)

    @staticmethod
    def LLM_worker(client:Client, model, command):
        """
        Description:
            Calls the LLM model and returns the response.
        Inputs:
            client: LLM client instance.
            model (str): Model name.
            command (list): List of messages for the LLM.
        Outputs:
            str: The LLM's response.
        """
        if model == "phi4:14b" or model == "Qwen3:30b":
            response = client.chat(model=model, messages=command, options={'temperature': 0.1})
            return response.message.content
        else:
            response = client.chat.completions.create(model="openai/gpt-4.1:floor", messages=command)
            return response.choices[0].message.content
    
    def send_to_LLM(self, conversation, extra_context):
        """
        Description:
            Sends the conversation and extra context to the LLM and returns the generated text.
        Inputs:
            conversation (list): List of conversation messages.
            extra_context (list): Additional context messages.
        Outputs:
            generated_text (str): Generated text from the LLM.
        """
        messages = [self.initial_prompt]
        messages.extend(extra_context)
        messages.extend(conversation)  
        generated_text = self.LLM_worker(self.client, self.model, messages)
        return generated_text

def load_configuration(config_file):
    """
    Description:
        Loads the configuration (model and initial prompt) from a YAML file.
    Inputs:
        config_file (str): Path to the YAML configuration file.
    Outputs:
        model (str): Model name.
        initial_prompt (str): Initial prompt string.
    """
    if not os.path.isfile(config_file):
        print(config_file + " does not exist!")
        raise FileExistsError
    else:
        print(f"Loading configuration from {config_file}...")
        config = yaml.load(
            open(config_file, "r", encoding="utf-8"),
            Loader=yamlloader.ordereddict.CLoader,
        )
        model = config["model"]
        initial_prompt = config["initial_prompt"]
    return model, initial_prompt

def extract_missions_and_drives(text):
    """
    Description:
        Extracts missions and drives from the LLM output text.
    Inputs:
        text (str or list): LLM output that contains missions and drives.
    Outputs:
        missions (list): List of [mission_tag, mission_value].
        drives (list): List of drive strings.
    """
    if isinstance(text, list):
        text = "\n".join(text)
    missions = []
    drives = []

    mission_blocks = re.findall(
        r'(Mission\d+:\s*\[.*?\]\s*Drive:\s*.*?)(?=(?:\n\s*\n|$))',
        text, re.DOTALL
    )

    for block in mission_blocks:
        tag_match = re.search(r'Mission\d+:\s*\[([^\],]+),\s*([0-9.]+)\]', block)
        drive_match = re.search(r'Drive:\s*(.*)', block)

        if tag_match and drive_match:
            mission_tag = tag_match.group(1)
            mission_value = float(tag_match.group(2))
            drive = drive_match.group(1).strip()

            missions.append([mission_tag, mission_value])
            drives.append(drive)

    return missions, drives

class ChatInterface:
    """
    Description:
        GUI chat interface to interact with the LLM.
    Inputs:
        root (tk.Tk): Tkinter root window.
        LLM (LLMmodel): LLM model instance.
    Outputs:
        None (GUI interaction and conversation state).
    """
    def __init__(self, root, LLM):
        self.root = root
        self.root.title("Chat Interface")
        self.LLM = LLM
        self.conversation = []
        self.final_purpose = None
        self.llm_mission = None
        self.LLM4drives = None
        self.drives = None
        self.awaiting_drive_feedback = False
        self.drive_conversation = []

        # Text area for conversation history
        self.chat_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, state='disabled', width=90, height=30, bg="#f4f4f4", font=("Arial", 11))
        self.chat_display.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        # Multiline input field
        self.user_input = tk.Text(root, width=70, height=4, font=("Arial", 12))
        self.user_input.grid(row=1, column=0, padx=10, pady=10)

        # Bind Ctrl+Enter to send message
        self.user_input.bind("<Control-Return>", self.send_message)

        # Send button
        self.send_button = tk.Button(root, text="Send", command=self.send_message, width=10, bg="#48C9F1", fg="white")
        self.send_button.grid(row=1, column=1, padx=10, pady=10)

        self.conversation = []

    def display_message(self, role, message):
        """
        Description:
            Displays a message in the chat history area.
        Inputs:
            role (str): 'user' or 'assistant'.
            message (str): Message content.
        Outputs:
            None (updates GUI).
        """
        self.chat_display.configure(state='normal')
        self.chat_display.insert(tk.END, f"{role.capitalize()}: {message}\n\n")
        self.chat_display.configure(state='disabled')
        self.chat_display.yview(tk.END)

    def send_message(self, event=None):
        """
        Description:
            Sends the user message, gets a reply from the LLM, and updates the chat.
        Inputs:
            event (tk.Event, optional): Event object from Tkinter.
        Outputs:
            None (updates conversation and GUI).
        """
        user_msg = self.user_input.get("1.0", tk.END).strip()
        if not user_msg:
            return

        self.conversation.append({'role': 'user', 'content': user_msg})
        self.display_message('user', user_msg)
        self.user_input.delete("1.0", tk.END)

        try:
            perceptions_file = os.path.join(current_dir, "objects.yaml")
            perceptions = yaml.load(
                open(perceptions_file, "r", encoding="utf-8"),
                Loader=yamlloader.ordereddict.CLoader,)
            objects = [{"role": "system", "content": str(perceptions["objects"])}]
            reply = self.LLM.send_to_LLM(self.conversation, objects)
            self.conversation.append({'role': 'assistant', 'content': reply})
            self.display_message('assistant', reply)
            save_conversation(self.conversation)

            if "Final description" in reply:
                self.final_purpose = reply
                messagebox.showinfo("Info", "Final description received. Generating missions...")
                self.generate_missions(objects)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def generate_missions(self, objects):
        """
        Description:
            Calls the LLM that generates missions based on the final purpose and objects.
        Inputs:
            objects (list): List of system context messages.
        Outputs:
            None (updates conversation and GUI).
        """
        missions_file = os.path.join(current_dir, "missions_prompt.yaml")
        needs_file = os.path.join(current_dir, "internal_needs.yaml")
        needs = yaml.load(
                open(needs_file, "r", encoding="utf-8"),
                Loader=yamlloader.ordereddict.CLoader,
            )
        needs_list = [{"role": "system", "content": str(needs["needs"])}]
        combined_data = [objects, needs_list]
        model, missions_prompt = load_configuration(missions_file)
        LLM4missions = LLMmodel(model, missions_prompt)

        try:
            mission_response = LLM4missions.send_to_LLM([{'role': 'user', 'content': self.final_purpose}], combined_data)
            self.llm_mission = mission_response
            self.conversation.append({'role': 'assistant', 'content': mission_response})
            self.display_message('assistant', mission_response)
            save_conversation(self.conversation)

            messagebox.showinfo("Info", "Missions generated. Generating drives...")
            self.generate_drives(objects)
        except Exception as e:
            messagebox.showerror("Mission Generation Error", str(e))

    def generate_drives(self, objects):
        """
        Description:
            Calls the LLM that generates drives based on the missions and objects.
        Parameters:
            objects (list): List of system context messages.
        Outputs:
            None (updates conversation and GUI).
        """
        drives_file = os.path.join(current_dir, "drives_prompt.yaml")
        model, drives_prompt = load_configuration(drives_file)
        LLM4drives = LLMmodel(model, drives_prompt)
        self.LLM4drives = LLM4drives
        first_prompt = self.final_purpose + "\n\n" + self.llm_mission
        self.drive_conversation.append({'role': 'user', 'content': first_prompt})
        try:
            drive_response = LLM4drives.send_to_LLM(self.drive_conversation, objects)
            self.conversation.append({'role': 'assistant', 'content': drive_response})
            self.drive_conversation.append({'role': 'assistant', 'content': drive_response})
            self.display_message('assistant', drive_response)
            save_conversation(self.conversation)

            if "Final drives" in drive_response:
                self.drives = drive_response
                messagebox.showinfo("Drives", "Final drives received. Interaction complete.")
                self.root.after(1000, self.root.destroy)
            else:
                self.awaiting_drive_feedback = True
                self.send_button.configure(command=lambda: self.send_drive_feedback(objects))
                self.user_input.bind('<Control-Return>', lambda event: self.send_drive_feedback(objects))
        except Exception as e:
            messagebox.showerror("Drives Generation Error", str(e))

    def send_drive_feedback(self, objects):
        """
        Description:
            Handles user feedback for drives if needed, sends it to the LLM, and updates the chat.
        Inputs:
            objects (list): List of system context messages.
        Outputs:
            None (updates conversation and GUI).
        """
        if not self.awaiting_drive_feedback:
            return

        feedback = self.user_input.get("1.0", tk.END).strip()
        if not feedback:
            return

        self.user_input.delete("1.0", tk.END)
        self.display_message('user', feedback)
        self.drive_conversation.append({'role': 'user', 'content': feedback})
        self.conversation.append({'role': 'user', 'content': feedback})

        try:
            drive_response = self.LLM4drives.send_to_LLM(self.drive_conversation, objects)
            self.drive_conversation.append({'role': 'assistant', 'content': drive_response})
            self.conversation.append({'role': 'assistant', 'content': drive_response})
            self.display_message('assistant', drive_response)
            save_conversation(self.conversation)

            if "Final answer" in drive_response:
                self.drives = drive_response
                self.awaiting_drive_feedback = False
                messagebox.showinfo("Drives", "Final drives received. Interaction complete.")
                self.root.after(10000, self.root.destroy)
            else:
                self.awaiting_drive_feedback = True
                self.send_button.configure(command=lambda: self.send_drive_feedback(objects))
                self.user_input.bind('<Control-Return>', lambda event: self.send_drive_feedback(objects))

        except Exception as e:
            messagebox.showerror("Feedback Error", str(e))

def interface():
    """
    Description:
        Main interface function to launch the chat, interact with the user, and extract missions and drives.
    Inputs:
        None
    Outputs:
        results.drives (str): Final drives string from the LLM.
        missions (list): List of [mission_tag, mission_value].
        drives (list): List of drive strings.
    """
    config_file = os.path.join(current_dir, "humanpurpose_prompt.yaml")
    init_conversation_file()
    model, initial_prompt = load_configuration(config_file)
    LLM = LLMmodel(model, initial_prompt)

    root = tk.Tk()
    results = ChatInterface(root, LLM)
    root.mainloop()
    missions, drives = extract_missions_and_drives(results.drives)
    return results.drives, missions, drives

if __name__ == "__main__":
    final_answer = interface()
    print(final_answer)
    missions, drives = extract_missions_and_drives(final_answer)

    print("MISSIONS:\n", missions)
    print("\nDRIVES:\n", drives)