import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
import logging
import speech_recognition as sr
import pyttsx3
import json
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeniusAI:
    def __init__(self, model_name='gpt2', max_length=150, num_return_sequences=1):
        self.model_name = model_name
        self.max_length = max_length
        self.num_return_sequences = num_return_sequences

        try:
            # Load pre-trained model and tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
            self.model.eval()  # Set the model to evaluation mode
            logger.info(f"Model and tokenizer for '{self.model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model or tokenizer: {e}")
            raise

    def generate_response(self, input_text):
        if not isinstance(input_text, str) or not input_text.strip():
            raise ValueError("Input text must be a non-empty string.")

        try:
            # Encode the input text
            input_ids = self.tokenizer.encode(input_text, return_tensors='pt')

            # Generate response
            with torch.no_grad():
                output = self.model.generate(
                    input_ids, 
                    max_length=self.max_length, 
                    num_return_sequences=self.num_return_sequences
                )

            # Decode the generated text
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def fine_tune_model(self, dataset_path):
        # Load dataset
        dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path=dataset_path,
            block_size=128
        )
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # Set up training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            overwrite_output_dir=True,
            num_train_epochs=3,  # Number of training epochs
            per_device_train_batch_size=2,  # Batch size per device during training
            save_steps=500,  # Number of steps before saving a checkpoint
            save_total_limit=2,  # Maximum number of checkpoints to keep
            logging_dir='./logs',  # Directory for storing logs
            logging_steps=100,  # Number of steps before logging training metrics
        )

        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
        )

        # Train the model
        trainer.train()

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            print("Recognizing...")
            text = recognizer.recognize_google(audio)
            print(f"You: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
            return None
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return None

def speak_text(text):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    # Set the voice to female
    for voice in voices:
        if 'female' in voice.name.lower():
            engine.setProperty('voice', voice.id)
            break
    engine.say(text)
    engine.runAndWait()

def save_interaction(user_input, ai_response, user_feedback):
    interaction = {
        "user_input": user_input,
        "ai_response": ai_response,
        "user_feedback": user_feedback
    }
    with open("interactions.json", "a") as f:
        f.write(json.dumps(interaction) + "\n")

def load_interactions(file_path):
    interactions = []
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            for line in f:
                interactions.append(json.loads(line))
    return interactions

def create_dataset_from_interactions(interactions, dataset_path):
    with open(dataset_path, "w") as f:
        for interaction in interactions:
            f.write(f"{interaction['user_input']}\n{interaction['user_feedback']}\n")

# Example usage
if __name__ == "__main__":
    genius_ai = GeniusAI()
    try:
        user_input = recognize_speech()
        if user_input:
            response = genius_ai.generate_response(user_input)
            print("Genius AI:", response)
            speak_text(response)

            # Collect user feedback
            print("Was the response satisfactory? (yes/no)")
            feedback = input().strip().lower()
            if feedback == "no":
                print("Please provide the correct response:")
                correct_response = input().strip()
                save_interaction(user_input, response, correct_response)
            else:
                save_interaction(user_input, response, response)

            # Periodically fine-tune the model
            interactions = load_interactions("interactions.json")
            create_dataset_from_interactions(interactions, "fine_tune_dataset.txt")
            genius_ai.fine_tune_model("fine_tune_dataset.txt")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
