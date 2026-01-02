"""
SMS Spam Classification System
A machine learning application for identifying spam messages using Perceptron algorithm
"""

import numpy as np
import pandas as pd
import gradio as gr
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ClassificationResult:
    """Container for classification results"""
    label: str
    confidence: float
    is_spam: bool
    timestamp: str
    message: str


@dataclass
class ModelMetrics:
    """Container for model performance metrics"""
    train_accuracy: float
    test_accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: np.ndarray


# ============================================================================
# Perceptron Classifier
# ============================================================================

class PerceptronClassifier:
    """
    Custom Perceptron implementation for binary classification
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iterations: int = 100,
        early_stopping: bool = True,
        patience: int = 15,
        random_state: int = 42
    ):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.early_stopping = early_stopping
        self.patience = patience
        self.random_state = random_state

        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.training_history: List[float] = []
        self._best_weights: Optional[np.ndarray] = None
        self._best_bias: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PerceptronClassifier':
        """Train the perceptron model"""
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape

        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0

        best_accuracy = 0.0
        patience_counter = 0

        for epoch in range(self.max_iterations):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for features, target in zip(X_shuffled, y_shuffled):
                linear_output = np.dot(features, self.weights) + self.bias
                prediction = 1 if linear_output >= 0 else 0

                update = self.learning_rate * (target - prediction)
                self.weights += update * features
                self.bias += update

            current_accuracy = accuracy_score(y, self.predict(X))
            self.training_history.append(current_accuracy)

            if self.early_stopping:
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    self._best_weights = self.weights.copy()
                    self._best_bias = self.bias
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"Early stopping triggered at epoch {epoch + 1}")
                        self.weights = self._best_weights
                        self.bias = self._best_bias
                        break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability estimates using sigmoid function"""
        linear_output = np.dot(X, self.weights) + self.bias
        clipped_output = np.clip(linear_output, -500, 500)
        probabilities = 1 / (1 + np.exp(-clipped_output))
        return probabilities


# ============================================================================
# SMS Classifier Application
# ============================================================================

class SMSClassifierApp:
    """Main application class for SMS spam classification"""

    def __init__(self):
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.model: Optional[PerceptronClassifier] = None
        self.metrics: Optional[ModelMetrics] = None
        self.classification_history: List[ClassificationResult] = []

    def load_and_prepare_data(self, url: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and preprocess the SMS dataset"""
        print("Loading dataset...")
        df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})

        self.vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            max_features=3000,
            strip_accents='unicode',
            lowercase=True,
            stop_words='english',
            min_df=2,
            max_df=0.95
        )

        X = self.vectorizer.fit_transform(df['message']).toarray()
        y = df['label'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"Dataset loaded: {len(df)} messages")
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")

        return X_train, X_test, y_train, y_test

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the perceptron classifier"""
        print("\nTraining perceptron model...")
        self.model = PerceptronClassifier(
            learning_rate=0.01,
            max_iterations=100,
            early_stopping=True,
            patience=15,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        print("Training completed successfully")

    def evaluate_model(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> ModelMetrics:
        """Evaluate model performance and store metrics"""
        train_predictions = self.model.predict(X_train)
        test_predictions = self.model.predict(X_test)

        self.metrics = ModelMetrics(
            train_accuracy=accuracy_score(y_train, train_predictions),
            test_accuracy=accuracy_score(y_test, test_predictions),
            precision=precision_score(y_test, test_predictions, zero_division=0),
            recall=recall_score(y_test, test_predictions, zero_division=0),
            f1_score=f1_score(y_test, test_predictions, zero_division=0),
            confusion_matrix=confusion_matrix(y_test, test_predictions)
        )

        self._print_evaluation_summary()
        return self.metrics

    def _print_evaluation_summary(self) -> None:
        """Print formatted evaluation metrics"""
        print(f"\n{'=' * 60}")
        print(f"{'MODEL PERFORMANCE SUMMARY':^60}")
        print(f"{'=' * 60}")
        print(f"Training Accuracy:    {self.metrics.train_accuracy:6.2%}")
        print(f"Test Accuracy:        {self.metrics.test_accuracy:6.2%}")
        print(f"Precision:            {self.metrics.precision:6.2%}")
        print(f"Recall:               {self.metrics.recall:6.2%}")
        print(f"F1 Score:             {self.metrics.f1_score:6.2%}")
        print(f"{'=' * 60}\n")

    def classify_message(self, message: str) -> Tuple[str, str]:
        """Classify a single SMS message"""
        if not message or not message.strip():
            return "", "Please enter a message to analyze."

        message_vector = self.vectorizer.transform([message]).toarray()
        prediction = self.model.predict(message_vector)[0]
        probability = self.model.predict_proba(message_vector)[0]

        is_spam = bool(prediction == 1)
        confidence = float(probability * 100 if is_spam else (1 - probability) * 100)

        result = ClassificationResult(
            label="SPAM" if is_spam else "HAM",
            confidence=round(confidence, 2),
            is_spam=is_spam,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            message=message
        )

        self.classification_history.append(result)

        # Create confidence bar
        confidence_bar = int(confidence)

        # Create status HTML
        if is_spam:
            status_html = f"""
            <div style="padding: 32px; background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); 
                        border-radius: 16px; color: white; box-shadow: 0 8px 16px rgba(239, 68, 68, 0.3);
                        border: 2px solid rgba(255, 255, 255, 0.1);">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
                    <h2 style="margin: 0; font-size: 32px; font-weight: 700;">Spam Detected</h2>
                    <div style="background: rgba(255, 255, 255, 0.2); padding: 8px 20px; border-radius: 24px;">
                        <span style="font-size: 24px; font-weight: 700;">{result.confidence:.1f}%</span>
                    </div>
                </div>
                <div style="background: rgba(255, 255, 255, 0.15); border-radius: 12px; height: 12px; overflow: hidden; margin-bottom: 20px;">
                    <div style="background: white; height: 100%; width: {confidence_bar}%; border-radius: 12px; 
                                box-shadow: 0 2px 4px rgba(0,0,0,0.2);"></div>
                </div>
                <p style="margin: 0; font-size: 16px; line-height: 1.6; opacity: 0.95;">
                    <strong>Warning:</strong> This message appears to be spam. Do not click any links, download attachments, 
                    or provide personal information.
                </p>
            </div>
            """
        else:
            status_html = f"""
            <div style="padding: 32px; background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                        border-radius: 16px; color: white; box-shadow: 0 8px 16px rgba(16, 185, 129, 0.3);
                        border: 2px solid rgba(255, 255, 255, 0.1);">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
                    <h2 style="margin: 0; font-size: 32px; font-weight: 700;">Legitimate Message</h2>
                    <div style="background: rgba(255, 255, 255, 0.2); padding: 8px 20px; border-radius: 24px;">
                        <span style="font-size: 24px; font-weight: 700;">{result.confidence:.1f}%</span>
                    </div>
                </div>
                <div style="background: rgba(255, 255, 255, 0.15); border-radius: 12px; height: 12px; overflow: hidden; margin-bottom: 20px;">
                    <div style="background: white; height: 100%; width: {confidence_bar}%; border-radius: 12px;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.2);"></div>
                </div>
                <p style="margin: 0; font-size: 16px; line-height: 1.6; opacity: 0.95;">
                    This message appears to be legitimate. However, always exercise caution with unexpected messages.
                </p>
            </div>
            """

        details = f"*Analyzed at {result.timestamp}*"

        return status_html, details

    def get_history(self) -> str:
        """Retrieve classification history"""
        if not self.classification_history:
            return "*No messages analyzed yet. Start by entering a message above.*"

        history_items = []
        for i, entry in enumerate(reversed(self.classification_history[-12:]), 1):
            preview = entry.message[:70] + "..." if len(entry.message) > 70 else entry.message
            status_badge = "🔴 SPAM" if entry.is_spam else "🟢 LEGITIMATE"

            history_items.append(
                f"**{i}. {status_badge}** — {entry.confidence:.1f}% confidence\n"
                f"*{entry.timestamp}*\n"
                f"> {preview}\n"
            )

        return "\n".join(history_items)

    def clear_history(self) -> str:
        """Clear classification history"""
        count = len(self.classification_history)
        if count == 0:
            return "*No history to clear.*"
        self.classification_history.clear()
        return f"*Successfully cleared {count} classification(s) from history.*"

    def get_quick_stats(self) -> str:
        """Get quick statistics"""
        if not self.classification_history:
            return "*No data available yet*"

        total = len(self.classification_history)
        spam_count = sum(1 for entry in self.classification_history if entry.is_spam)

        return f"**Total Analyzed:** {total} | **Spam:** {spam_count} | **Legitimate:** {total - spam_count}"


# ============================================================================
# Application Initialization
# ============================================================================

app_instance = SMSClassifierApp()

DATASET_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
X_train, X_test, y_train, y_test = app_instance.load_and_prepare_data(DATASET_URL)

app_instance.train_model(X_train, y_train)

metrics = app_instance.evaluate_model(X_train, X_test, y_train, y_test)


# ============================================================================
# Gradio Interface
# ============================================================================

def create_interface() -> gr.Blocks:
    """Create and configure the Gradio interface"""

    custom_css = """
    .message-input textarea {
        font-size: 16px !important;
        line-height: 1.7 !important;
        border-radius: 12px !important;
    }
    .metric-box {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        padding: 24px;
        border-radius: 16px;
        color: white;
        box-shadow: 0 8px 16px rgba(99, 102, 241, 0.3);
        border: 2px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.2s;
    }
    .metric-box:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(99, 102, 241, 0.4);
    }
    .metric-box h3 {
        margin: 0 0 12px 0;
        font-size: 13px;
        opacity: 0.9;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-box p {
        margin: 0;
        font-size: 36px;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .history-container {
        background: #ffffff;
        padding: 24px;
        border-radius: 16px;
        border: 2px solid #e5e7eb;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .main-container {
        max-width: 1400px;
        margin: 0 auto;
    }
    """

    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="slate",
            font=gr.themes.GoogleFont("Inter")
        ),
        css=custom_css,
        title="SMS Spam Classifier"
    ) as interface:

        with gr.Column(elem_classes="main-container"):
            # Header
            gr.Markdown("""
            # SMS Spam Classification System
            ### Intelligent message analysis powered by machine learning
            
            Paste any SMS message below to instantly determine if it's spam or legitimate.
            """)

            gr.Markdown("<br>")

            # Main Analysis Section
            with gr.Group():
                message_input = gr.Textbox(
                    label="Message to Analyze",
                    placeholder="Example: Congratulations! You've won a $1000 gift card. Click here to claim...",
                    lines=6,
                    max_lines=12,
                    elem_classes="message-input"
                )

                with gr.Row():
                    classify_button = gr.Button(
                        "Analyze Message",
                        variant="primary",
                        size="lg",
                        scale=4
                    )
                    clear_input_button = gr.Button("Clear", size="lg", scale=1, variant="secondary")

            gr.Markdown("<br>")

            # Results Section
            classification_output = gr.HTML()
            details_output = gr.Markdown()

            gr.Markdown("<br><br>")

            # Model Performance Metrics
            gr.Markdown("## Model Performance Metrics")
            gr.Markdown("*Trained on 5,572 real SMS messages*")

            gr.Markdown("<br>")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML(f"""
                    <div class="metric-box">
                        <h3>Accuracy</h3>
                        <p>{metrics.test_accuracy * 100:.1f}%</p>
                    </div>
                    """)

                with gr.Column(scale=1):
                    gr.HTML(f"""
                    <div class="metric-box">
                        <h3>Precision</h3>
                        <p>{metrics.precision * 100:.1f}%</p>
                    </div>
                    """)

                with gr.Column(scale=1):
                    gr.HTML(f"""
                    <div class="metric-box">
                        <h3>Recall</h3>
                        <p>{metrics.recall * 100:.1f}%</p>
                    </div>
                    """)

                with gr.Column(scale=1):
                    gr.HTML(f"""
                    <div class="metric-box">
                        <h3>F1 Score</h3>
                        <p>{metrics.f1_score * 100:.1f}%</p>
                    </div>
                    """)

            gr.Markdown("<br><br>")

            # History Section
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("## Analysis History")
                with gr.Column(scale=1):
                    stats_output = gr.Markdown()

            with gr.Group(elem_classes="history-container"):
                history_output = gr.Markdown()

                gr.Markdown("<br>")

                with gr.Row():
                    refresh_history_button = gr.Button("Refresh", size="sm", scale=1)
                    clear_history_button = gr.Button("Clear All History", size="sm", variant="secondary", scale=1)

            gr.Markdown("<br>")

            # Footer
            gr.Markdown("""
            ---
            <div style="text-align: center; color: #6b7280; font-size: 14px;">
                <p><strong>Disclaimer:</strong> This tool is for educational purposes. Always exercise caution with suspicious messages.</p>
                <p>Developed with Perceptron Algorithm | © 2025</p>
            </div>
            """)

        # Event Handlers
        def analyze_and_update(message):
            result, details = app_instance.classify_message(message)
            stats = app_instance.get_quick_stats()
            return result, details, stats

        classify_button.click(
            fn=analyze_and_update,
            inputs=message_input,
            outputs=[classification_output, details_output, stats_output]
        )

        clear_input_button.click(
            fn=lambda: ("", "", ""),
            outputs=[message_input, classification_output, details_output]
        )

        refresh_history_button.click(
            fn=lambda: (app_instance.get_history(), app_instance.get_quick_stats()),
            outputs=[history_output, stats_output]
        )

        clear_history_button.click(
            fn=lambda: (app_instance.clear_history(), app_instance.get_quick_stats()),
            outputs=[history_output, stats_output]
        )

        interface.load(
            fn=lambda: (app_instance.get_history(), app_instance.get_quick_stats()),
            outputs=[history_output, stats_output]
        )

    return interface


# ============================================================================
# Application Launch
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Launching SMS Spam Classification System...")
    print("=" * 60 + "\n")

    gradio_interface = create_interface()
    gradio_interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )