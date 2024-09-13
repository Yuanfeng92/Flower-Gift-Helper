import os
from flowerClassifier.components.predict_image import ImageClassifier
from flowerClassifier.utils.common import decodeImage
import gradio as gr

# Set environment variables
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

flower_info_dict = {
    "tulip": ["""Tulips 🌷 Perfect Love""", """- Meaning: Perfect, deep love
- Best for:
    - Partner: Spring celebrations, new beginnings
    - Friends: Congratulations on achievements
- Flower language: "Our love is as perfect as this bloom"
- Pro tip: Mix colors for a vibrant, playful bouquet"""],
    "sunflower": ["""Sunflowers 🌻 Radiant Joy""", """- Meaning: Adoration, loyalty, happiness
- Best for:
    - Friends: Birthdays, congratulations
    - Family: Housewarmings, get-well-soon
- Flower language: "You are the sunshine that brightens my day"
- Pro tip: Pair with bright, cheerful colors for an extra mood boost"""],
    "rose": ["""Roses 🌹 Ultimate Symbol of Love""", """- Meaning: Passionate romance, deep affection
- Best for:
    - Partner: Valentine's Day, anniversaries, romantic surprises
- Flower language: "My love for you burns with the intensity of a thousand suns"
- Pro tip: One rose says "I love you," a dozen amplifies the message"""],
    "daisy": ["Daisies 🌸 Cheerfulness", """- Meaning: Cheerfulness, innocence, purity
- Best for:
    - Siblings: Birthdays, congratulations
    - Colleagues: Work celebrations, thank you gifts
- Flower language: "Your smile lights up the room like these bright blooms"
- Pro tip: Available in many colors; choose recipient's favorite for a personal touch"""]
}


# ClientApp class definition
class ClientApp:
    def __init__(self):
        self.image = "inputImage.jpg"
        self.classifier = ImageClassifier()

clApp = ClientApp()

# Function to handle prediction using the Gradio interface
def predict(image):    
    # Prediction process
    result = clApp.classifier.predict(image)
    
    # Return the prediction result (for simplicity, returning the first result)
    return flower_info_dict[result][0], flower_info_dict[result][1]

# Gradio Interface
interface = gr.Interface(
    fn=predict,                            # Function to make predictions
    inputs=gr.Image(type="pil"),           # Input: An image in PIL format
    outputs=[gr.Label(label="Type of Flower"), gr.Textbox(label="More Infomation")],                        # Output: Prediction label (text)
    title="What Flower Is This?",              # Title of the interface
    description="Don't be a flower fool! Let me be your floral guru. Upload a flower image!"  # Simple instructions
)


# Launch the Gradio interface
if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=8080)  # Gradio will serve on port 8080