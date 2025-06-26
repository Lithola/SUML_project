import gradio as gr
import pandas as pd
import skops.io as sio

model_path = "../Model/steam_pipeline.skops"

# Wczytaj model z dopuszczeniem typu numpy.dtype
with open(model_path, "rb") as f:
    model = sio.load(f, trusted={"numpy.dtype"})

# Funkcja predykcyjna
def predict_price(english, required_age, categories, genres, platforms,
                  achievements, positive_ratings, negative_ratings, average_playtime):
    input_df = pd.DataFrame([{
        'english': int(english),
        'required_age': int(required_age),
        'categories': categories,
        'genres': genres,
        'platforms': platforms,
        'achievements': int(achievements),
        'positive_ratings': int(positive_ratings),
        'negative_ratings': int(negative_ratings),
        'average_playtime': float(average_playtime)
    }])
    prediction = model.predict(input_df)
    return f"Predicted Price: ${prediction[0]:.2f}"

# UI inputs
inputs = [
    gr.Radio(choices=["0", "1"], label="Is English?", value="1"),
    gr.Number(label="Required Age", value=0, precision=0),
    gr.Textbox(label="Categories (semicolon-separated)", value="Single-player;Partial Controller Support"),
    gr.Textbox(label="Genres", value="Action"),
    gr.Textbox(label="Platforms", value="windows"),
    gr.Number(label="Number of Achievements", value=0, precision=0),
    gr.Number(label="Positive Ratings", value=9817, precision=0),
    gr.Number(label="Negative Ratings", value=819, precision=0),
    gr.Number(label="Average Playtime (minutes)", value=209, precision=1),
]

# Przykładowe dane
examples = [
    ["1", 0, "Single-player;Partial Controller Support", "Action", "windows", 0, 9817, 819, 209],
    ["1", 18, "Single-player", "RPG", "windows;mac", 10, 24000, 1200, 340],
    ["0", 0, "Multiplayer", "Simulation", "linux", 0, 300, 50, 90]
]

# Gradio interfejs
app = gr.Interface(
    fn=predict_price,
    inputs=inputs,
    outputs="text",
    title="Steam Game Price Predictor 🎮",
    description="Enter game features to predict the Steam game price.",
    examples=examples,
    theme=gr.themes.Soft(),
    flagging_mode="never",
    article="© 2025 | Predictive model trained on Steam data."
)

if __name__ == "__main__":
    app.launch()
