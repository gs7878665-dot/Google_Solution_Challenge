# Fair Hire Intelligence Platform

An end-to-end AI bias auditing and mitigation platform designed to ensure hiring algorithms comply with the **EU AI Act (Article 10)**. This tool analyzes historical hiring datasets, detects proxy discrimination, mitigates bias through data transformation techniques, and leverages LLMs (Google Gemini) to explain fairness metrics in plain English.

## Features

1. **Bias Auditing**: Upload a CSV dataset, select a sensitive attribute (e.g., gender, race) and an outcome variable. The platform calculates:
   - **Disparate Impact Ratio**: Compares the success rate of minority vs. majority groups.
   - **Proxy Discrimination Flags**: Detects variables (like zip codes) that highly correlate with sensitive attributes.
   - **Correlation Heatmap & Representation**: Visualizes data distributions and variable correlations.

2. **Bias Mitigation Strategies**:
   - **Downsample Majority**: Reduces the size of the overrepresented group to match the minority.
   - **Reweight Samples**: Assigns higher importance weights to minority group samples during model training.
   - **Remove Proxy Columns**: Automatically drops features that are highly correlated with the sensitive attribute to prevent indirect bias.

3. **Accuracy vs. Fairness Trade-off**:
   Trains and compares a baseline Machine Learning model (Logistic Regression) against a "Fair" model post-mitigation. Visualizes the trade-off between predictive accuracy and the disparity ratio.

4. **AI Auditor Chatbot**:
   Integrates with Google's **Gemini API** to provide plain-English explanations of complex statistical metrics, helping HR managers understand the audit results and actionable fixes.

5. **EU AI Act Reporting**:
   Generates a downloadable, timestamped PDF report summarizing the disparate impact, representation, and compliance status.

## Technology Stack

### Backend
- **Python / Flask**: Serves the REST API endpoints.
- **Pandas & NumPy**: For efficient data processing, aggregation, and mathematical computations.
- **Scikit-Learn**: For training the Logistic Regression baseline and fair models, and tracking accuracy.
- **Google GenAI SDK**: Interfaces with Gemini (gemini-2.5-flash) to answer user queries about the data.
- **ReportLab**: Programmatically generates the PDF audit reports.
- **Python-Dotenv**: Manages the API key securely.

### Frontend
- **HTML/CSS/JavaScript**: Vanilla web stack with a modern, glassmorphic UI design.
- **Chart.js**: Renders interactive Bar, Doughnut, and Scatter charts.
- **PapaParse**: Parses CSV files on the client side to instantly populate form dropdowns before sending data to the server.

## Installation & Setup

1. **Clone the repository and navigate to the project directory:**
   ```bash
   cd main_sol
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the Environment:**
   - Ensure you have a `.env` file in the root directory.
   - Add your Google AI Studio API key to it:
     ```env
     GEMINI_API_KEY=your_google_api_key_here
     ```

4. **Run the Application:**
   ```bash
   python app.py
   ```

5. **Access the Platform:**
   Open your browser and navigate to `index.html` (you can open it directly or serve it via a local server like `http://127.0.0.1:5500/index.html`).

## How it Works (Code Overview)

- **`app.py`**: The main Flask server.
  - `/audit`: Reads the CSV, computes means per group, calculates disparate impact, generates a correlation matrix, and flags proxies (corr > 0.6).
  - `/fix`: Takes the requested strategy. Prepares the data, trains a baseline `LogisticRegression` model, applies the mitigation (downsampling, reweighting, or dropping proxies), trains a *new* model, and returns the comparative accuracy/disparity results along with a Base64-encoded "fixed" CSV.
  - `/explain`: Sends the computed metrics and the user's question to the `gemini-2.5-flash` model, requesting a plain-English explanation citing the EU AI Act.
  - `/report`: Uses ReportLab to draw a PDF canvas containing the audit metrics and a final compliance statement.

- **`index.html`**: The single-page application.
  - Handles the flow: Upload -> Audit Results -> Mitigation -> Chat/Report.
  - Uses modern CSS animations, Grid layouts, and custom chart rendering.

## Dataset
Use `sample.csv` to test the application. It contains hypothetical hiring data to demonstrate bias detection and mitigation.
