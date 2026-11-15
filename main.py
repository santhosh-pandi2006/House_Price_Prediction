"""Command-line house price prediction app."""

from pathlib import Path
import math
import pickle

import joblib
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "model" / "house_price_model.pkl"

# Keep this order aligned with the feature columns used to train the saved pipeline.
EXPECTED_FEATURES = [
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "parking",
    "prefarea",
    "furnishingstatus",
]

YES_NO_FEATURES = [
    ("mainroad", "Main road access"),
    ("guestroom", "Guest room"),
    ("basement", "Basement"),
    ("hotwaterheating", "Hot water heating"),
    ("airconditioning", "Air conditioning"),
    ("prefarea", "Preferred area"),
]
FURNISHING_OPTIONS = {"furnished", "semi-furnished", "unfurnished"}


def load_model():
    """Load the saved pipeline and check its expected input columns."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. "
            "Run the model development notebook first."
        )

    model = joblib.load(MODEL_PATH)
    model_features = list(getattr(model, "feature_names_in_", []))
    if model_features != EXPECTED_FEATURES:
        raise ValueError(
            "The saved model's feature columns do not match the application inputs. "
            f"Expected {EXPECTED_FEATURES}, but the model has {model_features}."
        )
    return model


def prompt_integer(label, minimum):
    """Ask again until the user enters a whole number in the allowed range."""
    while True:
        answer = input(f"{label}: ").strip()
        if not answer:
            print("This value is required.")
            continue

        try:
            value = int(answer)
        except ValueError:
            print("Please enter a whole number.")
            continue

        if value < minimum:
            print(f"Please enter a number greater than or equal to {minimum}.")
            continue
        return value


def prompt_yes_no(label):
    """Ask for a yes/no value and return the category expected by the model."""
    while True:
        answer = input(f"{label} (yes/no): ").strip().lower()
        if answer in {"yes", "y"}:
            return "yes"
        if answer in {"no", "n"}:
            return "no"
        print("Please enter yes or no.")


def prompt_furnishing_status():
    """Ask for one of the furnishing categories present in the dataset."""
    choices = ", ".join(sorted(FURNISHING_OPTIONS))
    while True:
        answer = input(f"Furnishing status ({choices}): ").strip().lower()
        if answer in FURNISHING_OPTIONS:
            return answer
        print(f"Please choose one of: {choices}.")


def collect_house_details():
    """Collect and validate each feature used by the trained model."""
    details = {
        "area": prompt_integer("Area", minimum=1),
        "bedrooms": prompt_integer("Bedrooms", minimum=1),
        "bathrooms": prompt_integer("Bathrooms", minimum=1),
        "stories": prompt_integer("Stories", minimum=1),
    }

    for feature_name, label in YES_NO_FEATURES[:4]:
        details[feature_name] = prompt_yes_no(label)

    details["parking"] = prompt_integer("Parking spaces", minimum=0)

    for feature_name, label in YES_NO_FEATURES[4:]:
        details[feature_name] = prompt_yes_no(label)

    details["furnishingstatus"] = prompt_furnishing_status()
    return details


def create_input_frame(details, model_features):
    """Create one Pandas row in the exact column order expected by the pipeline."""
    missing_features = [name for name in model_features if name not in details]
    extra_features = [name for name in details if name not in model_features]
    if missing_features or extra_features:
        raise ValueError(
            f"Input feature mismatch. Missing: {missing_features}; unexpected: {extra_features}."
        )

    input_frame = pd.DataFrame([details], columns=model_features)
    if input_frame.isna().any().any():
        raise ValueError("One or more required inputs are missing.")
    return input_frame


def main():
    """Run the interactive prediction flow."""
    try:
        model = load_model()
    except (
        AttributeError,
        EOFError,
        FileNotFoundError,
        ImportError,
        OSError,
        pickle.UnpicklingError,
        TypeError,
        ValueError,
    ) as error:
        print(f"Could not load the trained model: {error}")
        return

    print("House Price Prediction")
    print("Enter the house details below. Area and price use the dataset's units.")
    print("There is no location field in this dataset; 'Preferred area' is a yes/no feature.")

    try:
        details = collect_house_details()
        input_frame = create_input_frame(details, list(model.feature_names_in_))
    except (EOFError, KeyboardInterrupt):
        print("\nPrediction cancelled.")
        return
    except ValueError as error:
        print(f"Could not prepare the inputs: {error}")
        return

    try:
        predicted_price = float(model.predict(input_frame)[0])
        if not math.isfinite(predicted_price):
            raise ValueError("The model returned a non-finite prediction.")
    except (TypeError, ValueError, RuntimeError) as error:
        print(f"Could not generate a prediction: {error}")
        return

    print(f"\nEstimated house price: {predicted_price:,.2f} (dataset price units)")


if __name__ == "__main__":
    main()
