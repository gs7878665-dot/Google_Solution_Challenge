import io
import os
import json
import base64
from dotenv import load_dotenv

load_dotenv(override=True)
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from google import genai
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

app = Flask(__name__)
CORS(app)

# Gemini will automatically use GEMINI_API_KEY from environment

@app.route('/audit', methods=['POST'])
def audit():
    try:
        file = request.files['file']
        sensitive_col = request.form['sensitive_col']
        outcome_col = request.form['outcome_col']
        
        df = pd.read_csv(file)
        
        if sensitive_col not in df.columns or outcome_col not in df.columns:
            return jsonify({"error": "Columns not found in dataset"}), 400
            
        # Compute group representation
        df[sensitive_col] = df[sensitive_col].fillna("Unknown")
        representation = df[sensitive_col].value_counts().to_dict()
        
        # Compute group average outcomes
        # Ensure outcome_col is numeric
        if not pd.api.types.is_numeric_dtype(df[outcome_col]):
            return jsonify({"error": "Outcome column must be numeric"}), 400
            
        group_averages = df.groupby(sensitive_col)[outcome_col].mean().fillna(0).to_dict()
        
        # Calculate disparate impact ratio
        # Ratio: minority group mean / majority group mean
        groups = list(representation.keys())
        if len(groups) < 1:
            return jsonify({"error": "Need at least 1 group in sensitive attribute"}), 400
            
        # Determine majority and minority by count
        majority_group = max(representation, key=representation.get)
        # For simplicity in binary/multi-class, we take the smallest as minority
        minority_group = min(representation, key=representation.get)
        
        if majority_group == minority_group:
            if len(groups) > 1:
                # Tie, just pick first two
                majority_group = groups[0]
                minority_group = groups[1]
            else:
                majority_group = groups[0]
                minority_group = groups[0]
            
        majority_mean = group_averages[majority_group]
        minority_mean = group_averages[minority_group]
        
        disparate_impact = 1.0
        if majority_mean > 0:
            disparate_impact = float(minority_mean / majority_mean)
            
        bias_detected = disparate_impact < 0.8 or disparate_impact > 1.2
        
        # Compute correlation matrix across numeric columns
        # First encode sensitive_col if it's categorical to include it in proxy detection
        df_encoded = df.copy()
        if not pd.api.types.is_numeric_dtype(df_encoded[sensitive_col]):
            le = LabelEncoder()
            df_encoded[sensitive_col] = le.fit_transform(df_encoded[sensitive_col].astype(str))
            
        numeric_df = df_encoded.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr().replace({np.nan: None}).to_dict()
        
        proxy_flags = []
        if sensitive_col in corr_matrix:
            for col, corr_val in corr_matrix[sensitive_col].items():
                if col != sensitive_col and col != outcome_col and not pd.isna(corr_val):
                    if abs(corr_val) > 0.6:
                        proxy_flags.append({"column": col, "correlation": corr_val})
        
        return jsonify({
            "disparate_impact": disparate_impact,
            "group_averages": group_averages,
            "representation": representation,
            "correlation_matrix": corr_matrix,
            "proxy_flags": proxy_flags,
            "bias_detected": bias_detected,
            "majority_group": majority_group,
            "minority_group": minority_group
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def compute_disparate_impact(df, sensitive_col, outcome_col):
    if len(df) == 0:
        return 1.0
    representation = df[sensitive_col].value_counts()
    if len(representation) < 2:
        return 1.0
    majority_group = representation.idxmax()
    minority_group = representation.idxmin()
    group_averages = df.groupby(sensitive_col)[outcome_col].mean()
    if majority_group not in group_averages or minority_group not in group_averages:
        return 1.0
    maj_mean = group_averages[majority_group]
    min_mean = group_averages[minority_group]
    if maj_mean == 0:
        return 1.0
    return min_mean / maj_mean

@app.route('/fix', methods=['POST'])
def fix():
    try:
        file = request.files['file']
        strategy = request.form['strategy']
        sensitive_col = request.form['sensitive_col']
        outcome_col = request.form['outcome_col']
        
        df = pd.read_csv(file)
        original_df = df.copy()
        
        # Prepare data for modeling
        # We need to predict salary buckets based on experience
        # Assuming we create binary buckets for LogisticRegression (e.g., above/below median)
        median_outcome = df[outcome_col].median()
        y_raw = (df[outcome_col] >= median_outcome).astype(int)
        
        # Select numeric features for X
        X_raw = df.select_dtypes(include=[np.number]).drop(columns=[outcome_col], errors='ignore')
        if X_raw.empty:
            # Fallback: encode categorical features if no numeric features available
            for col in df.columns:
                if col != outcome_col:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
            X_raw = df.drop(columns=[outcome_col])
        
        # Encode sensitive col for fairness tracking if needed
        sensitive_series = df[sensitive_col]
        
        # Baseline model
        X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
            X_raw, y_raw, sensitive_series, test_size=0.2, random_state=42
        )
        
        baseline_clf = LogisticRegression(max_iter=1000)
        baseline_clf.fit(X_train, y_train)
        baseline_preds = baseline_clf.predict(X_test)
        baseline_accuracy = accuracy_score(y_test, baseline_preds)
        
        # Compute baseline disparity based on predictions
        # Use predicted values on test set to compute disparate impact
        test_df = X_test.copy()
        test_df[sensitive_col] = sens_test
        test_df['predicted_outcome'] = baseline_preds
        baseline_disparity = compute_disparate_impact(test_df, sensitive_col, 'predicted_outcome')
        
        # Apply fix strategy
        df_fixed = original_df.copy()
        sample_weights = None
        
        if strategy == "downsample":
            representation = df_fixed[sensitive_col].value_counts()
            if len(representation) > 1:
                minority_count = representation.min()
                downsampled_dfs = []
                for group in representation.index:
                    group_df = df_fixed[df_fixed[sensitive_col] == group]
                    downsampled_dfs.append(group_df.sample(n=minority_count, random_state=42))
                df_fixed = pd.concat(downsampled_dfs).reset_index(drop=True)
                
        elif strategy == "reweight":
            representation = df_fixed[sensitive_col].value_counts()
            total = len(df_fixed)
            weights = {group: total / (len(representation) * count) for group, count in representation.items()}
            sample_weights = df_fixed[sensitive_col].map(weights).values
            
        elif strategy == "remove_proxy":
            # Encode sensitive col to find correlation
            temp_df = df_fixed.copy()
            if not pd.api.types.is_numeric_dtype(temp_df[sensitive_col]):
                temp_df[sensitive_col] = LabelEncoder().fit_transform(temp_df[sensitive_col].astype(str))
            
            numeric_df = temp_df.select_dtypes(include=[np.number])
            corr_matrix = numeric_df.corr()
            cols_to_drop = []
            if sensitive_col in corr_matrix.columns:
                for col, corr_val in corr_matrix[sensitive_col].items():
                    if col != sensitive_col and col != outcome_col and not pd.isna(corr_val):
                        if abs(corr_val) > 0.6:
                            cols_to_drop.append(col)
                            
            if cols_to_drop:
                df_fixed = df_fixed.drop(columns=cols_to_drop, errors='ignore')
        
        # Fair model training
        y_fixed = (df_fixed[outcome_col] >= median_outcome).astype(int)
        X_fixed = df_fixed.select_dtypes(include=[np.number]).drop(columns=[outcome_col], errors='ignore')
        if X_fixed.empty:
            for col in df_fixed.columns:
                if col != outcome_col:
                    if not pd.api.types.is_numeric_dtype(df_fixed[col]):
                        df_fixed[col] = LabelEncoder().fit_transform(df_fixed[col].astype(str))
            X_fixed = df_fixed.drop(columns=[outcome_col])
            
        X_train_f, X_test_f, y_train_f, y_test_f, sens_train_f, sens_test_f = train_test_split(
            X_fixed, y_fixed, df_fixed[sensitive_col], test_size=0.2, random_state=42
        )
        
        fair_clf = LogisticRegression(max_iter=1000)
        
        if strategy == "reweight" and sample_weights is not None:
            # We need to split sample_weights too
            _, _, _, _, sw_train, _ = train_test_split(
                X_fixed, y_fixed, sample_weights, test_size=0.2, random_state=42
            )
            fair_clf.fit(X_train_f, y_train_f, sample_weight=sw_train)
        else:
            fair_clf.fit(X_train_f, y_train_f)
            
        fair_preds = fair_clf.predict(X_test_f)
        fair_accuracy = accuracy_score(y_test_f, fair_preds)
        
        test_df_f = X_test_f.copy()
        test_df_f[sensitive_col] = sens_test_f
        test_df_f['predicted_outcome'] = fair_preds
        fair_disparity = compute_disparate_impact(test_df_f, sensitive_col, 'predicted_outcome')
        
        # Encode fixed CSV to base64
        csv_buffer = io.StringIO()
        df_fixed.to_csv(csv_buffer, index=False)
        fixed_csv_b64 = base64.b64encode(csv_buffer.getvalue().encode('utf-8')).decode('utf-8')
        
        return jsonify({
            "baseline_accuracy": float(baseline_accuracy),
            "fair_accuracy": float(fair_accuracy),
            "baseline_disparity": float(baseline_disparity),
            "fair_disparity": float(fair_disparity),
            "fixed_csv": fixed_csv_b64
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/explain', methods=['POST'])
def explain():
    try:
        data = request.json
        metrics_json = data.get('metrics', '{}')
        question = data.get('question', '')
        
        prompt = f"""
You are a fairness auditor. Given these hiring bias metrics, answer the HR manager's question in plain English. Be specific with numbers. Cite EU AI Act Article 10 if bias is confirmed. Flag any proxy discrimination. Suggest one concrete fix. Keep response under 150 words.

Metrics:
{metrics_json}

Question:
{question}
"""
        client = genai.Client()
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        
        return jsonify({"answer": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/report', methods=['POST'])
def report():
    try:
        metrics = request.json
        
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        
        # Title
        c.setFont("Helvetica-Bold", 20)
        c.drawString(50, height - 50, "EU AI Act Fairness Audit Report")
        
        # Date
        import datetime
        c.setFont("Helvetica", 12)
        c.drawString(50, height - 75, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Disparate Impact
        y_pos = height - 120
        di = metrics.get('disparate_impact', 1.0)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_pos, "Disparate Impact Analysis")
        
        y_pos -= 25
        c.setFont("Helvetica", 12)
        c.drawString(50, y_pos, f"Disparate Impact Ratio: {di:.2f}x")
        
        y_pos -= 25
        bias_detected = metrics.get('bias_detected', False)
        status_text = "VIOLATES Article 10, requires mitigation under EU AI Act" if bias_detected else "COMPLIANT"
        if bias_detected:
            c.setFillColor(colors.red)
        else:
            c.setFillColor(colors.green)
        c.drawString(50, y_pos, f"Status: {status_text}")
        c.setFillColor(colors.black)
        
        # Representation
        y_pos -= 40
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_pos, "Group Representation")
        y_pos -= 25
        c.setFont("Helvetica", 12)
        representation = metrics.get('representation', {})
        for group, count in representation.items():
            c.drawString(70, y_pos, f"- {group}: {count}")
            y_pos -= 20
            
        # Proxy flags
        y_pos -= 20
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_pos, "Proxy Columns Detected")
        y_pos -= 25
        c.setFont("Helvetica", 12)
        proxy_flags = metrics.get('proxy_flags', [])
        if not proxy_flags:
            c.drawString(70, y_pos, "No proxy variables detected.")
            y_pos -= 20
        else:
            for pf in proxy_flags:
                c.drawString(70, y_pos, f"- {pf['column']} (Correlation: {pf['correlation']:.2f})")
                y_pos -= 20
        
        # Statement
        y_pos -= 40
        c.setFont("Helvetica-Oblique", 12)
        c.drawString(50, y_pos, f"Statement: Disparate impact {di:.2f}x - [{status_text}]")
        
        c.save()
        buffer.seek(0)
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name="Fairness_Audit_Report.pdf",
            mimetype="application/pdf"
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
