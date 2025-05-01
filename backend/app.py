import sklearn
import pandas as pd
import numpy as np  
import pymysql
import io
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import re
from flask import Flask, request, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from io import BytesIO
from datetime import timedelta
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from flask import request, redirect, url_for, render_template
from datetime import datetime

pymysql.install_as_MySQLdb()

app = Flask(__name__)

# Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Configurations
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/intellicleanse'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your_secret_key'  # Ensure a strong key
app.permanent_session_lifetime = timedelta(days=1)  # Extend session duration

app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)
# Initialize database
db = SQLAlchemy(app)

# User Table
class User(db.Model):
    __tablename__ = 'user'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(256), nullable=False)

    files = db.relationship('File', backref='uploader', lazy=True)
    team_collaborations = db.relationship('TeamCollaboration', backref='creator', lazy=True)

# File Table
class File(db.Model):
    __tablename__ = 'file'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(255), nullable=False)
    data = db.Column(db.LargeBinary, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)

    versions = db.relationship('FileVersion', backref='original_file', lazy=True)
    teams = db.relationship('TeamCollaboration', backref='target_file', lazy=True)

# Team Collaboration Table
class TeamCollaboration(db.Model):
    __tablename__ = 'team_collaboration'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    team_name = db.Column(db.String(100), nullable=False)
    no_of_members = db.Column(db.Integer, nullable=False)
    member_names = db.Column(db.Text, nullable=False)   # comma-separated names
    member_emails = db.Column(db.Text, nullable=False)  # comma-separated emails
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    file_id = db.Column(db.Integer, db.ForeignKey('file.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# File Versioning Table
class FileVersion(db.Model):
    __tablename__ = 'file_version'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    file_id = db.Column(db.Integer, db.ForeignKey('file.id'), nullable=False)
    version_data = db.Column(db.LargeBinary, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


@app.route('/profile', methods=['GET'])
def profile():
    if 'user_id' not in session:  # Check if user is logged in
        return jsonify({"error": "User not logged in"}), 401
    
    user = User.query.get(session['user_id'])
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    return jsonify({
        "username": user.username,
        "email": user.email
    })


@app.route('/edit_profile', methods=['GET', 'POST'])
def edit_profile():
    try:
        if 'user_id' not in session:
            return redirect(url_for('login'))  # Redirect to login if not logged in

        user = User.query.get(session['user_id'])
        if not user:
            return jsonify({"error": "User not found"}), 404

        if request.method == 'POST':
            username = request.form.get('username')
            email = request.form.get('email')

            if username:
                user.username = username
            if email:
                user.email = email

            db.session.commit()

            return redirect(url_for('profile'))  # Redirect back to profile page after saving changes

        return render_template('edit_profile.html', user=user)
    except Exception as e:
        print("Error in /edit_profile:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/signup', methods=['POST'])
def signup():
    try:
        data = request.get_json()
        print("Received signup data:", data)  # Debugging

        if not data:
            return jsonify({"error": "Invalid JSON data"}), 400

        username = data.get('username')
        email = data.get('email')
        password = data.get('password')

        if not username or not email or not password:
            return jsonify({"error": "Missing fields"}), 400

        user_exists = User.query.filter_by(email=email).first()
        if user_exists:
            return jsonify({"message": "Email already registered"}), 400

        hashed_password = generate_password_hash(password)
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        print("User created successfully:", email)  # Debugging
        return jsonify({"message": "Signup successful"}), 201
    except Exception as e:
        print("Error in /signup:", traceback.format_exc())  # Full error log
        return jsonify({"error": str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    try:
        # # session['user_id'] = user.id  # Assuming `user.id` is the ID of the logged-in user
        # session['user_id'] = user.id
        # session.permanent = True  # Keeps session active

        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON data"}), 400

        email = data.get('email')
        password = data.get('password')

        if not email or not password:
            return jsonify({"error": "Missing email or password"}), 400

        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password, password):
            return jsonify({"message": "Invalid credentials"}), 401

        session['user_id'] = user.id
        return jsonify({"message": "Login successful"}), 200
    except Exception as e:
        print("Error in /login:", traceback.format_exc())  # Full error log
        return jsonify({"error": str(e)}), 500

@app.route('/get_latest_file', methods=['GET'])
def get_latest_file():
    try:
        latest_file = File.query.order_by(File.id.desc()).first()
        if not latest_file:
            return jsonify({"error": "No files found in the database"}), 404

        filename = latest_file.name
        file_data = latest_file.data

        df = None
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_data))
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(file_data))
        else:
            return jsonify({"error": "Unsupported file format"}), 400

        preview_data = df.replace({pd.NA: None, float("nan"): None}).to_dict(orient='records')

        return jsonify({
            "message": f"Retrieved latest file '{filename}' successfully.",
            "preview": preview_data
        }), 200
    except Exception as e:
        print("Error in /get_latest_file:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

#function 1
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        filename = file.filename
        file_data = file.read()

        if not filename:
            return jsonify({"error": "Invalid file"}), 400
            

        # Save file to database
        new_file = File(name=filename, data=file_data)
        print("File object attributes:", new_file.__dict__)  # Debug
        db.session.add(new_file)
        db.session.commit()

        # Read the dataset
        file.seek(0)  # Reset file pointer
        df = None
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_data))
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(file_data))
        else:
            return jsonify({"error": "Unsupported file format"}), 400

        # Generate profiling report and handle NaN values
        profiling_info = {
            "columns": list(df.columns),
            "data_types": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicates": int(df.duplicated().sum()),
            "summary": df.describe(include='all').replace({pd.NA: None, float("nan"): None}).to_dict()
        }

        # Convert first few rows to JSON for preview (handle NaN values)
        preview_data = df.replace({pd.NA: None, float("nan"): None}).to_dict(orient='records')

        return jsonify({
            "message": f"File '{filename}' uploaded successfully and stored in database.",
            "preview": preview_data,
            "profiling": profiling_info
        }), 201
        return jsonify({ "file_id": str(file_id) })

    except Exception as e:
        print("Error in /upload:", traceback.format_exc())  # Detailed error log
        return jsonify({"error": str(e)}), 500


#function 2
@app.route('/clean_data', methods=['POST'])
def clean_data():
    try:
        latest_file = File.query.order_by(File.id.desc()).first()
        if not latest_file:
            return jsonify({"error": "No dataset found. Please upload a file first."}), 404

        filename = latest_file.name
        file_data = latest_file.data

        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_data))
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(file_data))
        else:
            return jsonify({"error": "Unsupported file format"}), 400

        # Remove duplicate rows and columns
        duplicates_before = df.duplicated().sum()
        df = df.drop_duplicates()
        df = df.loc[:, ~df.T.duplicated()]
        duplicates_after = df.duplicated().sum()

        # Handle missing values based on user choice
        missing_values_before = df.isnull().sum().to_dict()
        handle_missing = request.form.get('missing_values', 'mean')

        numeric_cols = df.select_dtypes(include=['number']).columns

        if handle_missing == 'mean':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif handle_missing == 'median':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif handle_missing == 'custom':
            custom_value = request.form.get('custom_value', '')
            df.fillna(custom_value, inplace=True)
        elif handle_missing == 'flag':
            df.fillna("MISSING", inplace=True)

        missing_values_after = df.isnull().sum().to_dict()

        # Convert all data types to JSON serializable formats
        cleaning_report = {
            "duplicates_removed": int(duplicates_before - duplicates_after),
            "missing_values_before": {k: int(v) for k, v in missing_values_before.items()},
            "missing_values_after": {k: int(v) for k, v in missing_values_after.items()}
        }

        # Convert DataFrame to JSON serializable format
        cleaned_data = df.replace({np.nan: None}).astype(object).to_dict(orient='records')

        return jsonify({
            "message": "Data cleaned successfully",
            "cleaning_report": cleaning_report,
            "cleaned_data": cleaned_data[:10]  # Preview first 10 rows
        })
    except Exception as e:
        print("Error in /clean_data:", traceback.format_exc())  # Detailed error logging
        return jsonify({"error": str(e)}), 500

#function 3
@app.route('/outlier_detection', methods=['POST'])
def outlier_detection():
    try:
        # Get the latest file uploaded
        latest_file = File.query.order_by(File.id.desc()).first()
        if not latest_file:
            return jsonify({"error": "No dataset found. Please upload a file first."}), 404

        filename = latest_file.name
        file_data = latest_file.data

        # Read the file content
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_data))
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(file_data))
        else:
            return jsonify({"error": "Unsupported file format"}), 400

        # Detect outliers using Z-score method
        numeric_cols = df.select_dtypes(include=['number']).columns
        df_zscore = df[numeric_cols].apply(zscore)
        outliers = (df_zscore.abs() > 3).sum(axis=0)  # Outliers if z-score > 3

        # Visualizations
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Histogram
        sns.histplot(df[numeric_cols[0]], kde=True, ax=axes[0])
        axes[0].set_title('Histogram of ' + numeric_cols[0])

        # Scatter Plot
        sns.scatterplot(data=df, x=numeric_cols[0], y=numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0], ax=axes[1])
        axes[1].set_title('Scatter Plot')

        # Box Plot
        sns.boxplot(data=df[numeric_cols[0]], ax=axes[2])
        axes[2].set_title('Box Plot of ' + numeric_cols[0])

        # Save the figure as a base64 string for the frontend
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        img_b64 = base64.b64encode(img.getvalue()).decode('utf-8')
        plt.close(fig)

        return jsonify({
            "message": "Outlier detection and visualizations generated successfully.",
            "outliers": outliers.to_dict(),
            "plot": img_b64
        })

    except Exception as e:
        print("Error in /outlier_detection:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

#function 4
@app.route('/data_standardization', methods=['POST'])
def standardize_data():
    try:
        latest_file = File.query.order_by(File.id.desc()).first()
        if not latest_file:
            return jsonify({"error": "No dataset found. Please upload a file first."}), 404

        filename = latest_file.name
        file_data = latest_file.data

        # Read the file content
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_data))
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(file_data))
        else:
            return jsonify({"error": "Unsupported file format"}), 400

        # Standardization Functions
        def standardize_text(col):
            return col.str.lower().str.strip()

        def standardize_date(col, date_format="%Y-%m-%d"):
            return pd.to_datetime(col, errors='coerce').dt.strftime(date_format)

        def standardize_numeric(col):
            return pd.to_numeric(col, errors='coerce')

        # Apply standardization rules
        for col in df.columns:
            if df[col].dtype == 'object':  # Text columns
                df[col] = standardize_text(df[col])
            elif df[col].dtype == 'datetime64[ns]':  # Date columns
                df[col] = standardize_date(df[col])
            elif df[col].dtype in ['int64', 'float64']:  # Numeric columns
                df[col] = standardize_numeric(df[col])

        # Convert DataFrame to JSON serializable format
        cleaned_data = df.replace({np.nan: None}).astype(object).to_dict(orient='records')

        return jsonify({
            "message": "Data standardized successfully.",
            "standardized_data": cleaned_data[:10]  # Preview first 10 rows
        })
    except Exception as e:
        print("Error in /data_standardization:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

#function 5 
@app.route('/datastandardization', methods=['GET'])
def data_standardization_page():
    return render_template('datastandardization.html')

#function 6 automated transformation 
@app.route('/transform_data', methods=['POST'])
def transform_data():
    try:
        latest_file = File.query.order_by(File.id.desc()).first()
        if not latest_file:
            return jsonify({"error": "No dataset found. Please upload a file first."}), 404

        filename = latest_file.name
        file_data = latest_file.data

        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_data))
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(file_data))
        else:
            return jsonify({"error": "Unsupported file format"}), 400

        # Ensure the dataset is not empty
        if df.empty:
            return jsonify({"error": "The uploaded file is empty."}), 400

        # Normalize numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            scaler = MinMaxScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        # Convert DataFrame to JSON format
        transformed_data = df.replace({np.nan: None}).astype(object).to_dict(orient='records')

        return jsonify({
            "message": "Data transformed successfully",
            "transformed_data": transformed_data[:10]  # Send only first 10 rows
        }), 200

    except Exception as e:
        print("Error in /transform_data:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# # balance check function
# @app.route('/check_balance', methods=['POST'])
# def check_balance():
#     try:
#         data = request.get_json()
#         file_id = data.get("file_id")

#         if not file_id:
#             return jsonify({"error": "No file_id provided in request."}), 400

#         file_record = File.query.get(file_id)
#         if not file_record:
#             return jsonify({"error": "Dataset not found with the given file_id."}), 404

#         filename = file_record.name
#         file_data = file_record.data

#         # Load file into DataFrame
#         if filename.endswith('.csv'):
#             df = pd.read_csv(io.BytesIO(file_data))
#         elif filename.endswith('.xlsx'):
#             df = pd.read_excel(io.BytesIO(file_data))
#         else:
#             return jsonify({"error": "Unsupported file format"}), 400

#         # Check if dataset is empty
#         if df.empty:
#             return jsonify({"error": "The uploaded file is empty."}), 400

#         # Detect target column (categorical or low unique values)
#         potential_targets = df.select_dtypes(include=['object', 'category']).columns.tolist()
#         if not potential_targets:
#             potential_targets = [col for col in df.columns if df[col].nunique() <= 10]

#         if not potential_targets:
#             return jsonify({"error": "No suitable target column found to check for balance."}), 400

#         target_col = potential_targets[-1]  # Choose the last one
#         class_counts = df[target_col].value_counts()
#         total = class_counts.sum()
#         balance_ratio = (class_counts / total).round(3).to_dict()

#         # Determine if dataset is balanced (no single class dominates > 70%)
#         is_balanced = all((count / total) <= 0.7 for count in class_counts)

#         return jsonify({
#             "message": "Balance check complete.",
#             "target_column": target_col,
#             "class_distribution": balance_ratio,
#             "is_balanced": is_balanced
#         })

#     except Exception as e:
#         print("Error in /check_balance:", traceback.format_exc())
#         return jsonify({"error": str(e)}), 500

@app.route('/save_collaboration', methods=['POST'])
def save_collaboration():
    try:
        team_name = request.form.get('team_name')
        no_of_members = int(request.form.get('no_of_members'))
        member_names = request.form.get('member_names')
        member_emails = request.form.get('member_emails')

        print("Received:", team_name, no_of_members, member_names, member_emails)

        # Use dummy user_id for now or fetch from session
        user_id = session.get('user_id', 1)
        latest_file = File.query.order_by(File.id.desc()).first()

        if not latest_file:
            return jsonify({"error": "No file found"}), 400

        new_team = TeamCollaboration(
            team_name=team_name,
            no_of_members=no_of_members,
            member_names=member_names,
            member_emails=member_emails,
            created_at=datetime.utcnow(),
            user_id=user_id,
            file_id=latest_file.id
        )

        db.session.add(new_team)
        db.session.commit()

        return jsonify({"message": "Team collaboration saved successfully!"}), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/collaboration_page', methods=['GET'])
def collaboration_page():
    success = request.args.get('success', 'false') == 'true'
    error = request.args.get('error', None)
    return render_template('collaboration.html', success=success, error=error)

# Add a route to serve collaborative.html for form access
@app.route('/collaborative_form', methods=['GET'])
def collaborative_form():
    return render_template('collaborative.html')

@app.route('/members')
def show_members_page():
    return render_template('members.html')  # ✅ renders from templates folder

@app.route('/api/members', methods=['GET'])
def get_team_members_api():
    teams = TeamCollaboration.query.all()
    data = []
    for team in teams:
        data.append({
            'team_name': team.team_name,
            'no_of_members': team.no_of_members,
            'member_names': team.member_names.split(','),
            'member_emails': team.member_emails.split(','),
            'created_at': team.created_at.strftime('%Y-%m-%d %H:%M'),
            'file_name': team.target_file.name  # 👈 Add file name here
        })
    return jsonify(data)


# Run Flask App
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
