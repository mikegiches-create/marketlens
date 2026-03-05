
import os
from dotenv import load_dotenv

# Set up backend directory path FIRST (before using it)
backend_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(backend_dir)

# Load the key from your .env file
load_dotenv()

# Load optional secret env file (gitignored)
secret_env = os.path.join(backend_dir, '.env.secret')
if os.path.exists(secret_env):
    load_dotenv(secret_env)

# Import Mistral AI Client
from mistral_client import MistralClientWrapper

# Initialize the Mistral AI client
client = MistralClientWrapper()

# Model configuration
MISTRAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
TEMPERATURE = 0.45

from flask import Flask, render_template, request, Response, jsonify, session, send_file, url_for, send_from_directory, redirect, flash
from markupsafe import Markup
from functools import wraps

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def role_required(*roles):
    """Decorator to require the current session role to be in `roles`."""
    def wrapper(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user' not in session:
                return redirect(url_for('login'))
            if roles and session.get('role') not in roles:
                return render_template('error.html', error='Access Denied', message="You don't have permission to access this page.")
            return f(*args, **kwargs)
        return decorated_function
    return wrapper

import time
import threading
import pandas as pd
from io import StringIO
import os
import json
import re
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

# scikit-learn - imported once at startup to catch broken builds early
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError as e:
    print("WARNING: scikit-learn not available:", e)
    print("Fix: pip uninstall scikit-learn -y && pip install scikit-learn --force-reinstall --no-cache-dir")
    SKLEARN_AVAILABLE = False
    class KMeans:
        def __init__(self, *a, **kw): pass
        def fit_predict(self, X): return [0] * len(X)
    class StandardScaler:
        def fit_transform(self, X): return X

# For charting
import plotly.express as px
from plotly.io import to_html

# LangChain Memory for conversation context (Customer Care)
class SimpleMemory:
    """Simple conversation memory to store chat history"""
    def __init__(self, memory_key="chat_history"):
        self.memory_key = memory_key
        self.messages = []

    def save_context(self, inputs, outputs):
        if isinstance(inputs, dict):
            user_msg = inputs.get("input", str(inputs))
        else:
            user_msg = str(inputs)

        if isinstance(outputs, dict):
            ai_msg = outputs.get("output", str(outputs))
        else:
            ai_msg = str(outputs)

        self.messages.append({"user": user_msg, "ai": ai_msg})

    def load_memory_variables(self, inputs=None):
        chat_history = ""
        for msg in self.messages[-5:]:
            if msg["user"]:
                chat_history += f"Customer: {msg['user']}\n"
            if msg["ai"]:
                chat_history += f"Assistant: {msg['ai']}\n\n"
        return {self.memory_key: chat_history}

# Your own module
from demographic import calculate_demographic_trends

# Set up paths for frontend templates and static files
# Explicitly set the folder paths so Vercel can find them
app = Flask(__name__, 
            template_folder='../frontend/templates',
            static_folder='../frontend/static')

app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# SECURITY CONFIGURATION - AI Safety Guardrails
DANGEROUS_KEYWORDS = [
    'delete', 'drop', 'truncate', 'alter', 'update', 'insert',
    'modify', 'remove', 'drop table', 'execute', 'exec',
    'import os', 'import sys', '__import__', 'eval', 'exec',
    'drop column', 'flush', 'purge', 'destroy', 'erase',
    'sys.', 'os.', 'subprocess', 'shell', 'lambda',
    'exec(', 'eval(', '__code__', 'globals()', 'locals()'
]

ANALYTICS_KEYWORDS = [
    'average', 'mean', 'median', 'sum', 'total', 'count', 'min', 'max',
    'trend', 'analyze', 'analysis', 'insight', 'pattern', 'distribution',
    'correlation', 'relationship', 'segment', 'group', 'filter', 'sort',
    'customer', 'data', 'metric', 'statistic', 'churn', 'revenue', 'spending',
    'income', 'age', 'purchase', 'order', 'sales', 'region', 'category',
    'how many', 'how much', 'what is', 'show me', 'tell me', 'list', 'find',
    'top', 'best', 'worst', 'highest', 'lowest', 'compare', 'difference'
]

GREETING_KEYWORDS = ['hello', 'hi', 'help', 'how', 'what', 'feature', 'platform', 'use']

# VISUALIZATION CONFIGURATION
VISUALIZATION_KEYWORDS = ['chart', 'graph', 'plot', 'visualize', 'visualization', 'show me', 'display', 'diagram']
VISUALIZATION_TYPES = {
    'pie': ['pie', 'distribution', 'breakdown', 'percentage', 'share'],
    'bar': ['bar', 'comparison', 'across', 'by', 'per'],
    'line': ['trend', 'over time', 'trend line', 'line chart', 'time series'],
    'scatter': ['scatter', 'correlation', 'relationship', 'scatter plot'],
    'histogram': ['histogram', 'frequency', 'histogram distribution'],
}

# LOCALIZATION CONFIGURATION
COMMON_REGIONS = {
    'nairobi': ['nairobi', 'nbi', 'cbd'],
    'mombasa': ['mombasa', 'msa', 'coastal'],
    'kisumu': ['kisumu', 'western'],
    'nakuru': ['nakuru', 'rift valley'],
    'kampala': ['kampala', 'uganda'],
    'dar es salaam': ['dar es salaam', 'dar', 'tanzania'],
    'accra': ['accra', 'ghana'],
    'lagos': ['lagos', 'nigeria'],
    'johannesburg': ['johannesburg', 'joburg', 'south africa'],
}

NON_TECHNICAL_TERMS = {
    'average': 'typical',
    'median': 'middle',
    'mean': 'average',
    'sum': 'total',
    'count': 'number of',
    'distribution': 'spread across',
    'correlation': 'connection between',
    'segment': 'group of',
    'cohort': 'group of',
    'metric': 'measurement',
    'outlier': 'unusual data point',
}

# Load configuration from environment variables
app.secret_key = os.getenv('SECRET_KEY', 'your-super-secret-key-change-this-in-production')
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', os.path.join(backend_dir, 'uploads'))
app.config['OUTPUT_FOLDER'] = os.getenv('OUTPUT_FOLDER', os.path.join(backend_dir, 'outputs'))
app.config['USERS_FILE'] = os.getenv('USERS_FILE', os.path.join(backend_dir, 'users.json'))
app.config['ALLOW_DEV_LOGIN'] = os.getenv('ALLOW_DEV_LOGIN', 'false').lower() == 'true'
app.config.setdefault('USERS', None)
uploaded_data = {}  # Store uploaded DataFrame by file_id

# --- MongoDB setup (pymongo) ---
MONGODB_URI = os.getenv('MONGODB_URI')
mongo_client = None
mongo_db = None
if MONGODB_URI:
    try:
        from pymongo import MongoClient
        mongo_client = MongoClient(MONGODB_URI)
        mongo_db = mongo_client.get_default_database() or mongo_client['app_db']
        print('Connected to MongoDB')
    except Exception as e:
        print(f'Warning: Could not connect to MongoDB: {e}')


def ensure_admin_user():
    """Create an admin user in MongoDB users collection if not present."""
    try:
        admin_username = os.getenv('ADMIN_USERNAME')
        admin_password = os.getenv('ADMIN_PASSWORD')
        admin_role = os.getenv('ADMIN_ROLE', 'CEO')
        if not (mongo_db and admin_username and admin_password):
            return
        users_col = mongo_db.get_collection('users')
        existing = users_col.find_one({'username': admin_username})
        if existing:
            print('Admin user already exists in MongoDB')
            return
        hashed = generate_password_hash(admin_password)
        users_col.insert_one({'username': admin_username, 'password': hashed, 'role': admin_role, 'created_at': datetime.utcnow()})
        print('Admin user created in MongoDB')
    except Exception as e:
        print(f'Failed to ensure admin user: {e}')

ensure_admin_user()

# Ensure upload and output folders exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder)

print("Template folder:", app.template_folder)
print("Available templates:", app.jinja_env.list_templates())

# DataFrame agents and conversation memories
dataframe_agents = {}
conversation_memories = {}


def get_or_create_memory(user_id):
    if user_id not in conversation_memories:
        conversation_memories[user_id] = SimpleMemory(memory_key="chat_history")
    return conversation_memories[user_id]


def create_agent_for_dataframe(df, file_id):
    try:
        dataframe_agents[file_id] = {"dataframe": df, "agent_type": "simple"}
        return {"dataframe": df, "agent_type": "simple"}
    except Exception as e:
        print(f"Error creating dataframe agent: {e}")
        return None


def query_dataframe_agent(query, file_id):
    try:
        if file_id not in dataframe_agents:
            if file_id in uploaded_data:
                df = uploaded_data[file_id]
                create_agent_for_dataframe(df, file_id)
            else:
                return {"error": "No data available. Please upload a CSV file first."}

        agent_info = dataframe_agents[file_id]
        df = agent_info["dataframe"]

        prompt = f"Analyze this dataframe based on the query. Query: {query}\n\nDataframe columns: {df.columns.tolist()}\n\nDataframe info:\n{df.describe().to_string()}"

        response = client.chat.completions.create(
            model=MISTRAL_MODEL,
            messages=[
                {"role": "system", "content": """You are a professional business data analyst having a conversation with a client.

CRITICAL RULES - YOU MUST FOLLOW THESE:
1. Write ONLY in plain text - NO markdown symbols ever
2. DO NOT use asterisks (**), hashtags (###), or dashes (-) for formatting
3. Write like you're speaking to someone in person
4. Use regular sentences and paragraphs
5. If you need to emphasize something, use CAPITAL LETTERS or write it clearly

Your tone should be:
- Professional but friendly (like a consultant)
- Clear and easy to understand
- Direct and helpful
- No technical jargon unless necessary

Format your response as simple paragraphs with blank lines between ideas."""},
                {"role": "user", "content": prompt}
            ],
            temperature=TEMPERATURE
        )

        output = response.choices[0].message.content
        output = output.replace('**', '').replace('###', '').replace('##', '').replace('#', '')
        output = re.sub(r'^\s*[-•]\s*', '', output, flags=re.MULTILINE)

        return {"result": output}
    except Exception as e:
        return {"error": f"Error processing query: {str(e)}"}


def detect_visualization_type(query):
    query_lower = query.lower()
    has_viz_request = any(kw in query_lower for kw in VISUALIZATION_KEYWORDS)
    if not has_viz_request:
        return None
    for chart_type, keywords in VISUALIZATION_TYPES.items():
        if any(kw in query_lower for kw in keywords):
            return chart_type
    return 'bar'


def generate_chart_from_dataframe(df, query, chart_type='bar'):
    try:
        query_lower = query.lower()
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        if not numeric_cols:
            return None

        primary_col = numeric_cols[0]
        group_col = None

        for col in df.columns:
            col_lower = col.lower()
            if col_lower in query_lower:
                if col in numeric_cols:
                    primary_col = col
                elif col in categorical_cols:
                    group_col = col

        if not group_col and categorical_cols:
            group_col = categorical_cols[0]

        if chart_type == 'pie' and group_col:
            grouped = df.groupby(group_col)[primary_col].sum().reset_index()
            fig = px.pie(grouped, values=primary_col, names=group_col,
                         title=f'{primary_col} Distribution by {group_col}')
        elif chart_type == 'bar' and group_col:
            grouped = df.groupby(group_col)[primary_col].agg(['mean', 'count']).reset_index()
            fig = px.bar(grouped, x=group_col, y='mean',
                         title=f'Average {primary_col} by {group_col}',
                         labels={'mean': f'Average {primary_col}'})
        elif chart_type == 'line' and group_col:
            grouped = df.groupby(group_col)[primary_col].mean().reset_index()
            fig = px.line(grouped, x=group_col, y=primary_col, markers=True,
                          title=f'{primary_col} Trend')
        elif chart_type == 'scatter' and len(numeric_cols) >= 2:
            fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                             title=f'{numeric_cols[0]} vs {numeric_cols[1]}')
        elif chart_type == 'histogram':
            fig = px.histogram(df, x=primary_col, title=f'Distribution of {primary_col}', nbins=20)
        else:
            if len(numeric_cols) >= 1:
                value_counts = df[numeric_cols[0]].value_counts().head(10).reset_index()
                fig = px.bar(value_counts, x=numeric_cols[0], y='count',
                             title=f'Top 10 {numeric_cols[0]} Values')
            else:
                return None

        return fig.to_json()
    except Exception as e:
        print(f"Error generating chart: {str(e)}")
        return None


def detect_region(query):
    query_lower = query.lower()
    for region, keywords in COMMON_REGIONS.items():
        for keyword in keywords:
            if keyword in query_lower:
                return region
    return None


def filter_dataframe_by_region(df, region):
    region_col = None
    for col in df.columns:
        if col.lower() in ['region', 'location', 'city', 'area']:
            region_col = col
            break
    if not region_col:
        return df
    filtered_df = df[df[region_col].str.lower().str.contains(region, case=False, na=False)]
    return filtered_df if len(filtered_df) > 0 else df


def translate_to_nontechnical(text):
    result = text
    for technical, simple in NON_TECHNICAL_TERMS.items():
        result = result.replace(technical, simple)
        result = result.replace(technical.capitalize(), simple.capitalize())
        result = result.replace(technical.upper(), simple.upper())
    return result


def format_business_summary(df, region=None, query=None):
    try:
        if region:
            df = filter_dataframe_by_region(df, region)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        summary_parts = []

        if region:
            summary_parts.append(f"📍 **{region.title()} Market Insights:**\n")
        if 'CustomerID' in df.columns:
            customer_count = df['CustomerID'].nunique()
            summary_parts.append(f"👥 **{customer_count:,} customers** in this area")
        if 'SalesAmount' in df.columns:
            total_sales = df['SalesAmount'].sum()
            avg_sales = df['SalesAmount'].mean()
            summary_parts.append(f"\n💰 **Sales Performance:**")
            summary_parts.append(f"   • Total revenue: KES {total_sales:,.0f}")
            summary_parts.append(f"   • Average per transaction: KES {avg_sales:,.0f}")
        if 'Spending_Score' in df.columns:
            avg_spending = df['Spending_Score'].mean()
            summary_parts.append(f"\n🛍️ **Customer Spending:** {avg_spending:.0f}/100 (higher = more active)")
        if 'Churn_Risk' in df.columns:
            high_risk = (df['Churn_Risk'] == 'High').sum()
            total = len(df)
            risk_pct = (high_risk / total) * 100 if total > 0 else 0
            summary_parts.append(f"\n⚠️ **At-Risk Customers:** {risk_pct:.1f}% need attention")

        return translate_to_nontechnical("\n".join(summary_parts))
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        return "Summary generation is not available."


def suggest_strategy_dynamically(df, query, ai_response):
    try:
        strategies = []
        query_lower = query.lower()

        if any(keyword in query_lower for keyword in ['at-risk', 'at risk', 'churn', 'leaving', 'retention']):
            if 'Churn_Risk' in df.columns:
                high_risk = df[df['Churn_Risk'] == 'High']
                high_risk_count = len(high_risk)
                if high_risk_count > 0:
                    risk_pct = (high_risk_count / len(df)) * 100
                    avg_value = high_risk['Annual_Income'].mean() if 'Annual_Income' in df.columns else 0
                    strategies.append(
                        f"🚨 **At-Risk Customer Alert:**\n"
                        f"• {high_risk_count} customers ({risk_pct:.1f}%) are at high churn risk\n"
                        f"• Average value of at-risk segment: ${avg_value:,.0f}\n"
                        f"\n💡 **Recommended Actions:**\n"
                        f"  1. Launch immediate outreach: personalized emails, loyalty offers, dedicated support\n"
                        f"  2. Offer special discounts or exclusive perks to re-engage them\n"
                        f"  3. Schedule check-in calls with your top at-risk customers\n"
                        f"  4. Create a win-back campaign with 30-day limited offers\n"
                        f"  5. Review why they're at risk: product issues? price concerns? poor service?"
                    )
            elif 'Spending_Score' in df.columns:
                low_spenders = df[df['Spending_Score'] < 30]
                if len(low_spenders) > 0:
                    strategies.append(
                        f"⚠️ **Low Engagement Alert:**\n"
                        f"• {len(low_spenders)} customers show low engagement patterns\n"
                        f"\n💡 **Re-engagement Strategy:**\n"
                        f"  1. Segment by reason: haven't purchased in 60+ days? Low purchase frequency?\n"
                        f"  2. Send re-engagement campaign: 'We miss you' offers, exclusive previews\n"
                        f"  3. Offer free shipping, bonus loyalty points, or early access to sales\n"
                        f"  4. Conduct quick survey: Why did they stop buying?"
                    )

        if any(keyword in query_lower for keyword in ['clv', 'lifetime value', 'customer value', 'high value']):
            if 'SalesAmount' in df.columns and 'CustomerID' in df.columns:
                total_clv = df['SalesAmount'].sum()
                avg_clv = df['SalesAmount'].mean()
                high_value = df[df['SalesAmount'] > df['SalesAmount'].quantile(0.75)]
                strategies.append(
                    f"💰 **Customer Lifetime Value (CLV) Explained:**\n"
                    f"• **What it means:** Total revenue each customer generates with you\n"
                    f"• Your total CLV: ${total_clv:,.0f}\n"
                    f"• Average CLV per customer: ${avg_clv:,.0f}\n"
                    f"• Top 25% of customers contribute: ${high_value['SalesAmount'].sum():,.0f}\n"
                    f"\n📊 **What This Tells You:**\n"
                    f"  • {len(high_value)} customers ({(len(high_value)/len(df)*100):.1f}%) drive ~{(high_value['SalesAmount'].sum()/total_clv*100):.0f}% of revenue\n"
                    f"\n🎯 **Next Steps:**\n"
                    f"  1. Visit CLV page to see detailed breakdown by customer segment\n"
                    f"  2. Identify your Champions (highest CLV) and VIPs\n"
                    f"  3. Create premium experiences for top customers\n"
                    f"  4. Use CLV to guide marketing budget allocation"
                )

        if any(keyword in query_lower for keyword in ['champion', 'loyal', 'rfm', 'segment', 'frequency']):
            if 'Spending_Score' in df.columns:
                champions = df[df['Spending_Score'] >= 75]
                loyal = df[(df['Spending_Score'] >= 50) & (df['Spending_Score'] < 75)]
                at_risk = df[df['Spending_Score'] < 30]
                strategies.append(
                    f"👑 **Customer Segment Strategy:**\n"
                    f"\n**Champions ({len(champions)} customers - {(len(champions)/len(df)*100):.0f}%):**\n"
                    f"  • Your best customers—they buy frequently and spend generously\n"
                    f"  • Strategy: VIP treatment, exclusive access, loyalty rewards\n"
                    f"\n**Loyal Customers ({len(loyal)} customers - {(len(loyal)/len(df)*100):.0f}%):**\n"
                    f"  • Regular buyers with consistent engagement\n"
                    f"  • Strategy: Keep them engaged and moving toward Champion status\n"
                    f"\n**At-Risk ({len(at_risk)} customers - {(len(at_risk)/len(df)*100):.0f}%):**\n"
                    f"  • Low engagement—they may be close to leaving\n"
                    f"  • Strategy: Win-back campaigns with special incentives"
                )

        if any(keyword in query_lower for keyword in ['revenue', 'sales', 'growth', 'increase', 'optimize']):
            if 'SalesAmount' in df.columns:
                total_revenue = df['SalesAmount'].sum()
                avg_transaction = df['SalesAmount'].mean()
                customer_count = df['CustomerID'].nunique() if 'CustomerID' in df.columns else len(df)
                strategies.append(
                    f"💹 **Revenue Optimization Strategy:**\n"
                    f"• Current total revenue: ${total_revenue:,.0f}\n"
                    f"• Average transaction value: ${avg_transaction:,.0f}\n"
                    f"• Customer count: {customer_count}\n"
                    f"\n**To grow revenue, focus on 3 areas:**\n"
                    f"  1. **Increase transaction value** (${avg_transaction:,.0f} → ${avg_transaction*1.2:,.0f}):\n"
                    f"     - Bundle products, upsell premium options\n"
                    f"  2. **Increase purchase frequency**:\n"
                    f"     - Email campaigns every 2-4 weeks\n"
                    f"     - Loyalty rewards for repeat purchases\n"
                    f"  3. **Acquire new customers**:\n"
                    f"     - Target lookalike audiences of your Champions"
                )

        if not strategies:
            if 'CustomerID' in df.columns and len(df) > 0:
                total_customers = df['CustomerID'].nunique()
                total_revenue = df['SalesAmount'].sum() if 'SalesAmount' in df.columns else 0
                strategies.append(
                    f"📈 **Dashboard Overview & Recommendations:**\n"
                    f"\n**Your Customer Base:**\n"
                    f"• Total customers: {total_customers}\n"
                    f"• Total transactions: {len(df)}\n"
                    f"• Total revenue: ${total_revenue:,.0f}\n"
                    f"\n**Explore Your Data:**\n"
                    f"  • Go to **RFM Analysis** to see customer segments\n"
                    f"  • Go to **CLV Analysis** to understand customer lifetime value\n"
                    f"  • Go to **Behavior** to see purchase patterns\n"
                    f"  • Go to **Churn Analysis** to identify customers likely to leave"
                )

        return "\n\n".join(strategies) if strategies else ""
    except Exception as e:
        print(f"Error suggesting strategy: {str(e)}")
        return ""


# Upload progress tracking
upload_progress = {}


def track_upload(file_id):
    for percent in range(0, 101, 10):
        if file_id in upload_progress:
            upload_progress[file_id] = percent
            time.sleep(0.5)
    if file_id in upload_progress:
        del upload_progress[file_id]


def generate_progress(file_id):
    while file_id in upload_progress:
        percent = upload_progress.get(file_id, 0)
        print(f"Sending progress: {percent}% for file_id: {file_id}")
        yield f"data: {percent}\n\n"
        time.sleep(0.5)
    print(f"Completed upload for file_id: {file_id}")
    yield "data: 100\n\n"
    yield "event: complete\ndata: Upload complete!\n\n"


# ─────────────────────────────────────────────
# AUTH ROUTES
# ─────────────────────────────────────────────

@app.route('/login', methods=['GET', 'POST'])
def login():
    try:
        def load_users():
            users = app.config.get('USERS')
            if users is None:
                users = {}
                try:
                    if os.path.exists(app.config['USERS_FILE']):
                        with open(app.config['USERS_FILE'], 'r', encoding='utf-8') as f:
                            users = json.load(f)
                except Exception:
                    users = {}
                app.config['USERS'] = users
            return app.config['USERS']

        error = None

        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            role = request.form.get('role')

            if not username or not password or not role:
                return render_template('login.html', error='Please fill all fields.')

            users = load_users()

            if username not in users:
                return render_template('login.html', error='Invalid username or password.')

            user_data = users[username]

            if not check_password_hash(user_data['password'], password):
                return render_template('login.html', error='Invalid username or password.')

            if user_data.get('role') != role:
                return render_template('login.html', error='Invalid role selected for this user.')

            session['user'] = username
            session['role'] = user_data.get('role', 'Company Analyst')
            session['email'] = user_data.get('email', '')
            session['business_name'] = user_data.get('business_name', '')
            session['business_category'] = user_data.get('business_category', '')
            session['login_time'] = datetime.now().isoformat()

            return redirect(url_for('home'))

        return render_template('login.html', error=None)
    except Exception as e:
        return render_template('login.html', error=f'An unexpected error occurred: {str(e)}')


@app.route('/logout')
def logout():
    try:
        session.clear()
        return redirect(url_for('login'))
    except Exception as e:
        return render_template('error.html', error='Logout Error', message=f"An error occurred during logout: {str(e)}")


@app.route('/api/get_business_info/<username>', methods=['GET'])
def get_business_info(username):
    try:
        users = {}
        if os.path.exists(app.config['USERS_FILE']):
            with open(app.config['USERS_FILE'], 'r', encoding='utf-8') as f:
                users = json.load(f)
        if username in users:
            user_data = users[username]
            return jsonify({'success': True, 'business_name': user_data.get('business_name', ''), 'business_category': user_data.get('business_category', '')})
        else:
            return jsonify({'success': False, 'message': 'User not found'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/settings')
@login_required
def settings():
    try:
        return render_template('settings.html')
    except Exception as e:
        return render_template('error.html', error='Settings Error', message=f"An error occurred loading settings: {str(e)}")


@app.route('/change_password', methods=['POST'])
@login_required
def change_password():
    def load_users():
        users = {}
        try:
            if os.path.exists(app.config['USERS_FILE']):
                with open(app.config['USERS_FILE'], 'r', encoding='utf-8') as f:
                    users = json.load(f)
        except Exception:
            pass
        return users

    def save_users(users):
        try:
            with open(app.config['USERS_FILE'], 'w', encoding='utf-8') as f:
                json.dump(users, f, indent=2)
        except Exception:
            pass

    current_password = request.form.get('current_password')
    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_password')
    username = session.get('user')

    if not current_password or not new_password or not confirm_password:
        flash('All fields are required.', 'danger')
        return redirect(url_for('settings'))

    if new_password != confirm_password:
        flash('New passwords do not match.', 'danger')
        return redirect(url_for('settings'))

    if len(new_password) < 6:
        flash('Password must be at least 6 characters long.', 'danger')
        return redirect(url_for('settings'))

    users = load_users()
    if username not in users:
        flash('User not found.', 'danger')
        return redirect(url_for('settings'))

    if not check_password_hash(users[username]['password'], current_password):
        flash('Current password is incorrect.', 'danger')
        return redirect(url_for('settings'))

    users[username]['password'] = generate_password_hash(new_password)
    save_users(users)
    flash('Password updated successfully!', 'success')
    return redirect(url_for('settings'))


@app.route('/delete_account')
@login_required
def delete_account():
    def load_users():
        users = {}
        try:
            if os.path.exists(app.config['USERS_FILE']):
                with open(app.config['USERS_FILE'], 'r', encoding='utf-8') as f:
                    users = json.load(f)
        except Exception:
            pass
        return users

    def save_users(users):
        try:
            with open(app.config['USERS_FILE'], 'w', encoding='utf-8') as f:
                json.dump(users, f, indent=2)
        except Exception:
            pass

    username = session.get('user')
    users = load_users()
    if username in users:
        del users[username]
        save_users(users)
    session.clear()
    flash('Your account has been deleted.', 'success')
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    try:
        def load_users():
            users = app.config.get('USERS')
            if users is None:
                users = {}
                try:
                    if os.path.exists(app.config['USERS_FILE']):
                        with open(app.config['USERS_FILE'], 'r', encoding='utf-8') as f:
                            users = json.load(f)
                except Exception:
                    users = {}
                app.config['USERS'] = users
            return app.config['USERS']

        def save_users(users):
            try:
                with open(app.config['USERS_FILE'], 'w', encoding='utf-8') as f:
                    json.dump(users, f, indent=2)
            except Exception:
                pass

        if request.method == 'POST':
            username = request.form.get('username')
            email = request.form.get('email')
            password = request.form.get('password')
            role = request.form.get('role')
            business_name = request.form.get('business_name')
            business_category = request.form.get('business_category')

            if not username or not email or not password or not role or not business_name or not business_category:
                return render_template('register.html', error='Please fill all fields.')

            if role not in ('CEO', 'Company Analyst'):
                return render_template('register.html', error='Invalid role selected.')

            valid_categories = [
                'Retail', 'E-commerce', 'Technology', 'Healthcare', 'Finance',
                'Education', 'Hospitality', 'Real Estate', 'Manufacturing',
                'Professional Services', 'Food & Beverage', 'Transportation',
                'Media', 'Telecommunications', 'Other'
            ]
            if business_category not in valid_categories:
                return render_template('register.html', error='Invalid business category selected.')

            users = load_users()
            if username in users:
                return render_template('register.html', error='Username already exists. Choose another.')

            hashed = generate_password_hash(password)
            users[username] = {
                'password': hashed,
                'role': role,
                'email': email,
                'business_name': business_name,
                'business_category': business_category
            }
            save_users(users)
            flash('Account created successfully! Please login with your credentials.', 'success')
            return redirect(url_for('login'))

        return render_template('register.html', error=None)
    except Exception as e:
        return render_template('register.html', error=f'An unexpected error occurred: {str(e)}')


# ─────────────────────────────────────────────
# PASSWORD RESET ROUTES
# ─────────────────────────────────────────────

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    def load_users():
        users = app.config.get('USERS')
        if users is None:
            users = {}
            try:
                if os.path.exists(app.config['USERS_FILE']):
                    with open(app.config['USERS_FILE'], 'r', encoding='utf-8') as f:
                        users = json.load(f)
            except Exception:
                users = {}
            app.config['USERS'] = users
        return app.config['USERS']

    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')

        if not username or not email:
            flash('Please fill all fields.', 'danger')
            return render_template('forgot_password.html')

        users = load_users()
        if username not in users:
            flash('User not found. Please check your username.', 'danger')
            return render_template('forgot_password.html')

        user_data = users[username]
        if 'email' in user_data and user_data['email'].lower() != email.lower():
            flash('Email does not match our records.', 'danger')
            return render_template('forgot_password.html')

        flash(f'Password reset instructions: Please contact your administrator or use the password reset page.', 'info')
        reset_link = url_for('reset_password', _external=False)
        flash(Markup(f'Click <a href="{reset_link}" style="color: #166534; font-weight: 600; text-decoration: underline;">here to reset your password</a> directly.'), 'success')
        return render_template('forgot_password.html')

    return render_template('forgot_password.html')


@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    def load_users():
        users = app.config.get('USERS')
        if users is None:
            users = {}
            try:
                if os.path.exists(app.config['USERS_FILE']):
                    with open(app.config['USERS_FILE'], 'r', encoding='utf-8') as f:
                        users = json.load(f)
            except Exception:
                users = {}
            app.config['USERS'] = users
        return app.config['USERS']

    def save_users(users):
        try:
            with open(app.config['USERS_FILE'], 'w', encoding='utf-8') as f:
                json.dump(users, f, indent=2)
            app.config['USERS'] = users
        except Exception:
            pass

    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')

        if not username or not email or not new_password or not confirm_password:
            flash('Please fill all fields.', 'danger')
            return render_template('reset_password.html')

        if new_password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('reset_password.html')

        users = load_users()
        if username not in users:
            flash('User not found.', 'danger')
            return render_template('reset_password.html')

        user_data = users[username]
        if 'email' in user_data and user_data['email'].lower() != email.lower():
            flash('Email does not match our records.', 'danger')
            return render_template('reset_password.html')

        user_data['password'] = generate_password_hash(new_password)
        users[username] = user_data
        save_users(users)
        flash('Password successfully reset! You can now login with your new password.', 'success')
        return redirect(url_for('login'))

    return render_template('reset_password.html')


# ─────────────────────────────────────────────
# AI / CHAT ROUTES
# ─────────────────────────────────────────────

@app.route('/ask_data', methods=['POST'])
@login_required
def ask_data():
    try:
        data = request.get_json()
        query = data.get('query', '').strip()

        if not query:
            return jsonify({"error": "Please provide a query."})

        query_lower = query.lower()
        for keyword in DANGEROUS_KEYWORDS:
            if keyword in query_lower:
                return jsonify({"error": "❌ Security Policy Violation: Data modification operations are not allowed."})

        file_id = session.get('last_uploaded_file_id')
        if not file_id:
            return jsonify({"error": "No data uploaded. Please upload a CSV file first."})

        enhanced_query = f"""
IMPORTANT: You have READ-ONLY access to this dataframe. You cannot modify, delete, or change any data.

User Query: {query}

Constraints:
1. Only perform read-only operations
2. Do NOT attempt to modify the dataframe
3. Provide analysis, insights, and summaries

Proceed with analysis:
"""
        response = query_dataframe_agent(enhanced_query, file_id)
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"})


@app.route('/ask_data_ui')
@login_required
def ask_data_ui():
    try:
        if 'last_uploaded_file_id' not in session:
            return render_template('ask_data.html', error="No data uploaded. Please upload a CSV file first.")
        return render_template('ask_data.html')
    except Exception as e:
        return render_template('error.html', error='Ask Data Error', message=f"An error occurred loading the ask data page: {str(e)}")


@app.route('/chat', methods=['POST'])
@login_required
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        user_id = session.get('user')

        if not user_message:
            return jsonify({"error": "Please provide a message."})

        memory = get_or_create_memory(user_id)

        user_message_lower = user_message.lower()
        for keyword in DANGEROUS_KEYWORDS:
            if keyword in user_message_lower:
                return jsonify({
                    "response": "❌ I cannot perform data modification operations. I have read-only access to your data.",
                    "has_data": False,
                    "security_filtered": True
                })

        has_analytics_keyword = any(keyword in user_message_lower for keyword in ANALYTICS_KEYWORDS)
        if not has_analytics_keyword:
            has_greeting = any(keyword in user_message_lower for keyword in GREETING_KEYWORDS)
            if not has_greeting:
                return jsonify({
                    "response": "ℹ️ I'm specialized in customer data analysis. Please ask me about your customer data!",
                    "has_data": False,
                    "out_of_scope": True
                })

        file_id = session.get('last_uploaded_file_id')
        has_data = file_id and file_id in uploaded_data

        context = ""
        if has_data:
            df = uploaded_data[file_id]
            detected_region = detect_region(user_message)
            df_context = filter_dataframe_by_region(df, detected_region) if detected_region else df
            context = f"\n\nUser has uploaded a dataset with {len(df_context)} rows and columns: {', '.join(df_context.columns.tolist())}"
            context += f"\nData summary:\n{df_context.describe().to_string()}"
            context += f"\n\nBusiness Summary:\n{format_business_summary(df_context, detected_region, user_message)}"

        system_prompt = """You are the Smart Segmentation Support Assistant, a polite and professional customer care representative.

IMPORTANT FORMATTING RULES:
• Write in natural, conversational language - NO markdown formatting
• DO NOT use asterisks (**) for bold text
• DO NOT use hashtags (###) for headers
• Use plain text only

TONE & PERSONALITY:
• Always be courteous, patient, and helpful
• Adopt a warm, consultative approach

SAFETY & BOUNDARIES:
• Never perform Write, Delete, or Update operations
• You are designed for analysis and support only"""

        chat_history = memory.load_memory_variables({}).get("chat_history", "")

        response = client.chat.completions.create(
            model=MISTRAL_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{context}\n\n{chat_history}\n\nUser message: {user_message}"}
            ],
            temperature=TEMPERATURE
        )

        ai_response = response.choices[0].message.content
        ai_response = ai_response.replace('**', '').replace('###', '').replace('##', '').replace('#', '')
        ai_response = re.sub(r'^\s*[-•]\s*', '', ai_response, flags=re.MULTILINE)

        memory.save_context({"input": user_message}, {"output": ai_response})

        if has_data:
            try:
                df = uploaded_data[file_id]
                suggested_strategy = suggest_strategy_dynamically(df, user_message, ai_response)
                if suggested_strategy:
                    ai_response = f"{ai_response}\n\n{suggested_strategy}"
            except Exception as e:
                print(f"⚠️ Could not generate strategy suggestion: {str(e)}")

        chart_json = None
        chart_type = None
        if has_data:
            chart_type = detect_visualization_type(user_message)
            if chart_type:
                try:
                    df = uploaded_data[file_id]
                    region = detect_region(user_message)
                    if region:
                        df = filter_dataframe_by_region(df, region)
                    chart_json = generate_chart_from_dataframe(df, user_message, chart_type)
                except Exception as e:
                    print(f"⚠️ Could not generate chart: {str(e)}")

        response_data = {"response": ai_response, "has_data": has_data}
        if chart_json:
            response_data["chart"] = chart_json
            response_data["chart_type"] = chart_type

        return jsonify(response_data)
    except Exception as e:
        print(f"Chat error: {str(e)}")
        return jsonify({"error": f"Error processing message: {str(e)}"})


# ─────────────────────────────────────────────
# CORE APP ROUTES
# ─────────────────────────────────────────────

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv'}


@app.route('/', methods=['GET', 'POST'])
def home():
    try:
        if 'user' not in session:
            return redirect(url_for('login'))

        if request.method == 'POST':
            if 'file' not in request.files:
                return render_template('index.html', error='No file part')
            file = request.files['file']
            if file.filename == '':
                return render_template('index.html', error='No selected file')
            if file and allowed_file(file.filename):
                file_id = id(file)
                upload_progress[file_id] = 0
                threading.Thread(target=track_upload, args=(file_id,), daemon=True).start()
                try:
                    df = pd.read_csv(file)
                    uploaded_data[file_id] = df
                    session['last_uploaded_file_id'] = file_id
                    agent = create_agent_for_dataframe(df, file_id)
                    if agent:
                        print(f"✅ Dataframe agent created for file_id: {file_id}")
                except Exception as e:
                    return render_template('index.html', error=f"Error processing file: {str(e)}")
                return Response(generate_progress(file_id), mimetype='text/event-stream')

        return render_template('index.html')
    except Exception as e:
        return render_template('index.html', error=f"An unexpected error occurred: {str(e)}")


@app.route('/dashboard')
def dashboard():
    try:
        if 'user' not in session:
            return redirect(url_for('login'))

        role = session.get('role')
        view = request.args.get('view', 'analyst' if role == 'Company Analyst' else 'executive')
        context = {
            'user': session.get('user'),
            'role': role,
            'login_time': session.get('login_time'),
            'view': view
        }
        return render_template('dashboard.html', **context)
    except Exception as e:
        return render_template('error.html', error='Dashboard Error', message=f"An error occurred loading the dashboard: {str(e)}")


# ─────────────────────────────────────────────
# OVERVIEW ROUTE — TABLE VIEW
# ─────────────────────────────────────────────

@app.route('/overview')
@login_required
def overview_analysis():
    try:
        error = None
        summary = None
        table_html = None
        columns = []
        month_list = []
        selected_month = request.args.get('month', 'All')

        if 'last_uploaded_file_id' in session and session['last_uploaded_file_id'] in uploaded_data:
            df = uploaded_data[session['last_uploaded_file_id']].copy()
            print(f"Columns available in overview: {df.columns.tolist()}")

            # ── Month filtering ─────────────────────────────────────────────
            time_column = next((c for c in ['OrderDate', 'PurchaseDate', 'Date'] if c in df.columns), None)
            if time_column:
                df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
                df.dropna(subset=[time_column], inplace=True)
                df['_Month'] = df[time_column].dt.to_period('M').astype(str)
                month_list = sorted(df['_Month'].unique().tolist())
                if selected_month != 'All':
                    df = df[df['_Month'] == selected_month]
                # drop helper column before rendering
                df = df.drop(columns=['_Month'])

            # ── Summary stat cards ──────────────────────────────────────────
            summary = {
                'total_customers': df['CustomerID'].nunique() if 'CustomerID' in df.columns else len(df),
                'avg_spending':    round(df['Spending_Score'].mean(), 2)  if 'Spending_Score' in df.columns else 'N/A',
                'avg_income':      round(df['Annual_Income'].mean(), 2)   if 'Annual_Income'  in df.columns else 'N/A',
                'most_common_region': (
                    df['Region'].mode()[0]
                    if 'Region' in df.columns and not df['Region'].dropna().empty
                    else None
                ),
            }

            # ── Raw data table (DataTables) ─────────────────────────────────
            columns = df.columns.tolist()
            table_html = df.to_html(
                classes='table table-bordered table-striped table-hover display',
                index=True,
                table_id='overviewTable',
                border=0
            )

        else:
            error = "No data available. Please upload a CSV file first."
            print(f"Error in overview: {error}")

        return render_template(
            'overview.html',
            error=error,
            summary=summary,
            table_html=table_html,
            columns=columns,
            month_list=month_list,
            selected_month=selected_month
        )
    except Exception as e:
        return render_template('error.html', error='Overview Analysis Error', message=f"An error occurred in overview analysis: {str(e)}")


# ─────────────────────────────────────────────
# ANALYSIS ROUTES
# ─────────────────────────────────────────────

@app.route('/rfm', methods=['GET', 'POST'])
@login_required
def rfm_analysis():
    error = None
    pie_chart = None
    bar_chart = None
    table = None
    download_link = None
    selected_segment = request.values.get('segment', 'All')
    analysis_date_str = request.form.get('analysis_date') or request.args.get('analysis_date')
    analysis_date = pd.to_datetime(analysis_date_str) if analysis_date_str else None

    if 'last_uploaded_file_id' not in session or session['last_uploaded_file_id'] not in uploaded_data:
        error = "No data available. Please upload a CSV file first."
        return render_template('rfm.html', error=error, selected_segment=selected_segment, analysis_date=analysis_date_str)

    df = uploaded_data[session['last_uploaded_file_id']]
    print(f"Columns available in rfm: {df.columns.tolist()}")

    if request.method == 'POST' and analysis_date:
        required_columns = ['CustomerID', 'OrderDate', 'Annual_Income']
        if not all(col in df.columns for col in required_columns):
            error = "Uploaded CSV must contain CustomerID, OrderDate, and Annual_Income columns."
            return render_template('rfm.html', error=error, selected_segment=selected_segment, analysis_date=analysis_date_str)

        df['OrderDate'] = pd.to_datetime(df['OrderDate'])
        current_date = analysis_date

        rfm_table = df.groupby(['CustomerID']).agg({
            'OrderDate': lambda x: (current_date - x.max()).days,
            'CustomerID': 'count',
            'Annual_Income': 'sum'
        }).rename(columns={'OrderDate': 'Recency', 'CustomerID': 'Frequency', 'Annual_Income': 'Monetary'})

        try:
            rfm_table['R_Score'] = pd.qcut(rfm_table['Recency'], 4, labels=[4, 3, 2, 1], duplicates='drop')
        except ValueError:
            rfm_table['R_Score'] = 2

        try:
            rfm_table['F_Score'] = pd.qcut(rfm_table['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4], duplicates='drop')
        except ValueError:
            rfm_table['F_Score'] = 2

        try:
            rfm_table['M_Score'] = pd.qcut(rfm_table['Monetary'].rank(method='first'), 4, labels=[1, 2, 3, 4], duplicates='drop')
        except ValueError:
            rfm_table['M_Score'] = 2

        rfm_table['RFM_Score'] = rfm_table['R_Score'].astype(str) + rfm_table['F_Score'].astype(str) + rfm_table['M_Score'].astype(str)

        def segment_name(score):
            if score == '444':
                return 'Champions'
            elif score[0] == '4':
                return 'Loyal Customers'
            elif score[1] == '4':
                return 'Frequent Buyers'
            elif score[2] == '4':
                return 'Big Spenders'
            elif score[0] == '1':
                return 'At Risk'
            else:
                return 'Others'

        rfm_table['Segment'] = rfm_table['RFM_Score'].apply(segment_name)

        if 'FirstName' in df.columns and 'LastName' in df.columns:
            df['CustomerName'] = df['FirstName'] + ' ' + df['LastName']
        elif 'FirstName' in df.columns:
            df['CustomerName'] = df['FirstName']

        if 'CustomerName' in df.columns:
            customer_names = df[['CustomerID', 'CustomerName']].drop_duplicates().set_index('CustomerID')
            rfm_table = rfm_table.merge(customer_names, left_index=True, right_index=True, how='left')
        else:
            rfm_table['CustomerName'] = 'N/A'


        clustering_data = rfm_table[['Recency', 'Frequency', 'Monetary']].dropna()
        if not clustering_data.empty:
            scaler = StandardScaler()
            rfm_scaled = scaler.fit_transform(clustering_data)
            kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
            rfm_table.loc[clustering_data.index, 'Cluster'] = kmeans.fit_predict(rfm_scaled)
        else:
            rfm_table['Cluster'] = None

        rfm_table = rfm_table[['CustomerName', 'Recency', 'Frequency', 'Monetary', 'RFM_Score', 'Segment', 'Cluster']]

        if selected_segment != 'All':
            rfm_table = rfm_table[rfm_table['Segment'] == selected_segment]

        csv_path = os.path.join(app.static_folder, 'rfm_output.csv')
        rfm_table.to_csv(csv_path)
        download_link = url_for('static', filename='rfm_output.csv')

        table = rfm_table.to_html(classes='table table-bordered table-striped table-hover display', index=True, table_id="rfmTable")
        table += '''
        <script src="https://cdn.datatables.net/1.10.24/js/jquery.dataTables.min.js"></script>
        <script>$(document).ready(function() { $('#rfmTable').DataTable(); });</script>
        '''

        segment_counts = rfm_table['Segment'].value_counts().sort_values(ascending=False)
        pie_chart_data = {
            'data': [{'values': segment_counts.values.tolist(), 'labels': segment_counts.index.tolist(), 'type': 'pie'}],
            'layout': {'title': 'RFM Segment Distribution'}
        }
        pie_chart = f'<div id="pie-chart" style="width: 100%; height: 400px;"></div><script src="https://cdn.plot.ly/plotly-latest.min.js"></script><script>Plotly.newPlot("pie-chart", {json.dumps(pie_chart_data)});</script>'

        avg_monetary = [rfm_table[rfm_table['Segment'] == seg]['Monetary'].mean() for seg in segment_counts.index]
        bar_chart_data = {
            'data': [{'x': segment_counts.index.tolist(), 'y': avg_monetary, 'type': 'bar', 'text': [f"${val:.2f}" for val in avg_monetary], 'textposition': 'auto'}],
            'layout': {'title': 'Average Monetary by Segment', 'yaxis': {'title': 'Average Monetary ($)'}}
        }
        bar_chart = f'<div id="bar-chart" style="width: 100%; height: 400px;"></div><script>Plotly.newPlot("bar-chart", {json.dumps(bar_chart_data)});</script>'

    elif not analysis_date:
        error = "Please select an analysis date and submit the form."

    return render_template('rfm.html', error=error, pie_chart=pie_chart, bar_chart=bar_chart, table=table, download_link=download_link, selected_segment=selected_segment, analysis_date=analysis_date_str)


@app.route('/clv')
@login_required
def clv_analysis():
    error = None
    pie_chart = None
    bar_chart = None
    line_chart = None
    scatter_plot = None
    clv_results = None
    clv_summary = None
    download_link = None

    file_id = session.get('last_uploaded_file_id')
    if not file_id or file_id not in uploaded_data:
        error = "No data available. Please upload a CSV file first."
        return render_template('clv.html', error=error, pie_chart=pie_chart, bar_chart=bar_chart, line_chart=line_chart, scatter_plot=scatter_plot, clv_results=clv_results, clv_summary=clv_summary, download_link=download_link)

    df = uploaded_data[file_id]
    if not all(col in df.columns for col in ['CustomerID', 'OrderDate', 'SalesAmount']):
        error = "CSV must contain 'CustomerID', 'OrderDate', and 'SalesAmount' columns."
        return render_template('clv.html', error=error, pie_chart=pie_chart, bar_chart=bar_chart, line_chart=line_chart, scatter_plot=scatter_plot, clv_results=clv_results, clv_summary=clv_summary, download_link=download_link)

    df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    df['OrderMonth'] = df['OrderDate'].dt.to_period('M').astype(str)

    clv_table = df.groupby('CustomerID').agg({'SalesAmount': 'sum', 'OrderDate': 'count'}).rename(columns={'SalesAmount': 'CLV', 'OrderDate': 'OrderCount'}).reset_index()
    clv_table['CLV_Quartile'] = pd.qcut(clv_table['CLV'], 4, labels=['Q1 - Lowest', 'Q2 - Low-Mid', 'Q3 - High-Mid', 'Q4 - Highest'])


    scaler = StandardScaler()
    features = scaler.fit_transform(clv_table[['CLV', 'OrderCount']])
    kmeans = KMeans(n_clusters=4, random_state=0)
    clv_table['CLV_Cluster'] = kmeans.fit_predict(features)

    clv_brackets = pd.cut(clv_table['CLV'], bins=[0, 200, 500, 1000, 2000, float('inf')], labels=['<200', '200-500', '500-1000', '1000-2000', '2000+'])
    pie_counts = clv_brackets.value_counts().sort_index()
    pie_data = {'data': [{'labels': pie_counts.index.tolist(), 'values': pie_counts.values.tolist(), 'type': 'pie'}], 'layout': {'title': 'CLV Bracket Distribution'}}
    pie_chart = f'<div id="clv-pie"></div><script src="https://cdn.plot.ly/plotly-latest.min.js"></script><script>Plotly.newPlot("clv-pie", {json.dumps(pie_data)});</script>'

    top_customers = clv_table.nlargest(10, 'CLV')
    bar_data = {'data': [{'x': top_customers['CustomerID'].astype(str).tolist(), 'y': top_customers['CLV'].tolist(), 'type': 'bar', 'text': [f"${v:.2f}" for v in top_customers['CLV']], 'textposition': 'auto'}], 'layout': {'title': 'Top 10 Customers by CLV', 'yaxis': {'title': 'CLV ($)'}}}
    bar_chart = f'<div id="clv-bar"></div><script>Plotly.newPlot("clv-bar", {json.dumps(bar_data)});</script>'

    clv_monthly = df.groupby('OrderMonth')['SalesAmount'].sum().sort_index()
    line_data = {'data': [{'x': clv_monthly.index.tolist(), 'y': clv_monthly.values.tolist(), 'mode': 'lines+markers', 'line': {'shape': 'linear'}}], 'layout': {'title': 'CLV Accumulation Over Time', 'xaxis': {'title': 'Month'}, 'yaxis': {'title': 'Total Sales'}}}
    line_chart = f'<div id="clv-line"></div><script>Plotly.newPlot("clv-line", {json.dumps(line_data)});</script>'

    scatter_data = {'data': [{'x': clv_table['OrderCount'].tolist(), 'y': clv_table['CLV'].tolist(), 'mode': 'markers', 'type': 'scatter', 'marker': {'size': 10, 'color': clv_table['CLV_Cluster'].tolist(), 'colorscale': 'Viridis', 'showscale': True}}], 'layout': {'title': 'CLV vs Frequency', 'xaxis': {'title': 'Order Count'}, 'yaxis': {'title': 'CLV'}}}
    scatter_plot = f'<div id="clv-scatter"></div><script>Plotly.newPlot("clv-scatter", {json.dumps(scatter_data)});</script>'

    clv_summary = {"Total Customers": len(clv_table), "Total CLV": f"${clv_table['CLV'].sum():,.2f}", "Average CLV": f"${clv_table['CLV'].mean():,.2f}", "Max CLV": f"${clv_table['CLV'].max():,.2f}", "Min CLV": f"${clv_table['CLV'].min():,.2f}"}

    selected_segment = request.args.get('segment', 'All')
    if selected_segment != 'All':
        clv_table = clv_table[clv_table['CLV_Quartile'] == selected_segment]

    clv_results = clv_table.sort_values(by='CLV', ascending=False)
    output_file = f"clv_results_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_file)
    clv_results.to_csv(output_path, index=False)
    download_link = url_for('download_results', filename=output_file)

    return render_template('clv.html', error=error, pie_chart=pie_chart, bar_chart=bar_chart, line_chart=line_chart, scatter_plot=scatter_plot, clv_results=clv_results, clv_summary=clv_summary, download_link=download_link)


@app.route('/churn')
@login_required
def churn_analysis():
    error = None
    churn_pie_chart = None
    churn_bar_chart = None
    churn_line_chart = None
    churn_summary = None
    churn_results = pd.DataFrame()
    download_link = None

    file_id = session.get('last_uploaded_file_id')
    if file_id and file_id in uploaded_data:
        df = uploaded_data[file_id]
        if 'Spending_Score' in df.columns and 'CustomerID' in df.columns:
            df['Churn_Risk'] = pd.cut(df['Spending_Score'], bins=[0, 30, 70, 100], labels=['High', 'Medium', 'Low'])
            selected_risk = request.args.get('risk', 'All')
            if selected_risk != 'All':
                df = df[df['Churn_Risk'] == selected_risk]

            churn_counts = df['Churn_Risk'].value_counts().sort_index()
            pie_data = {'data': [{'labels': churn_counts.index.tolist(), 'values': churn_counts.values.tolist(), 'type': 'pie'}], 'layout': {'title': 'Churn Risk Distribution'}}
            churn_pie_chart = f'<div id="churn-pie"></div><script src="https://cdn.plot.ly/plotly-latest.min.js"></script><script>Plotly.newPlot("churn-pie", {json.dumps(pie_data)});</script>'

            avg_spending = df.groupby('Churn_Risk')['Spending_Score'].mean()
            bar_data = {'data': [{'x': avg_spending.index.tolist(), 'y': avg_spending.values.tolist(), 'type': 'bar', 'text': [f"{v:.1f}" for v in avg_spending], 'textposition': 'auto'}], 'layout': {'title': 'Avg Spending by Churn Risk', 'yaxis': {'title': 'Avg Spending Score'}}}
            churn_bar_chart = f'<div id="churn-bar"></div><script>Plotly.newPlot("churn-bar", {json.dumps(bar_data)});</script>'

            sorted_df = df.sort_values(by='CustomerID')
            line_data = {'data': [{'x': sorted_df['CustomerID'].astype(str).tolist(), 'y': sorted_df['Spending_Score'].tolist(), 'mode': 'lines+markers'}], 'layout': {'title': 'Spending Score Trend by Customer', 'xaxis': {'title': 'CustomerID'}, 'yaxis': {'title': 'Spending Score'}}}
            churn_line_chart = f'<div id="churn-line"></div><script>Plotly.newPlot("churn-line", {json.dumps(line_data)});</script>'

            churn_summary = {'Total Customers': len(df), 'High Risk': int((df['Churn_Risk'] == 'High').sum()), 'Medium Risk': int((df['Churn_Risk'] == 'Medium').sum()), 'Low Risk': int((df['Churn_Risk'] == 'Low').sum())}

            cols = ['CustomerID', 'Spending_Score', 'Churn_Risk']
            if 'CustomerName' in df.columns:
                cols.insert(1, 'CustomerName')
            churn_results = df[cols].copy()
            churn_results.columns = churn_results.columns.str.replace('_', ' ')

            output_file = f"churn_results_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_file)
            churn_results.to_csv(output_path, index=False)
            download_link = url_for('download_results', filename=output_file)
        else:
            error = "CSV must contain 'CustomerID' and 'Spending_Score' columns."
    else:
        error = "No data available. Please upload a CSV file first."

    return render_template('churn.html', error=error, churn_pie_chart=churn_pie_chart, churn_bar_chart=churn_bar_chart, churn_line_chart=churn_line_chart, churn_summary=churn_summary, churn_results=churn_results, download_link=download_link)


@app.route('/behavior', methods=['GET', 'POST'])
def behavior_analysis():
    error = None
    pie_chart = None
    bar_chart = None
    line_chart = None
    table = None
    download_link = None
    selected_segment = request.values.get('segment', 'All')

    if 'last_uploaded_file_id' not in session or session['last_uploaded_file_id'] not in uploaded_data:
        error = "No data available. Please upload a CSV file first."
        return render_template('behavior.html', error=error, selected_segment=selected_segment)

    df = uploaded_data[session['last_uploaded_file_id']]
    if not all(col in df.columns for col in ['CustomerID', 'OrderDate', 'OrderCount']):
        error = "CSV must contain 'CustomerID', 'OrderDate', and 'OrderCount' columns."
        return render_template('behavior.html', error=error, selected_segment=selected_segment)

    df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    df['OrderMonth'] = df['OrderDate'].dt.to_period('M').astype(str)

    behavior_table = df.groupby('CustomerID').agg({'OrderCount': 'sum'}).rename(columns={'OrderCount': 'TotalOrders'}).reset_index()
    behavior_table['BehaviorType'] = behavior_table['TotalOrders'].apply(lambda x: 'Frequent Buyer' if x >= 5 else 'Two-Time Buyer' if x >= 2 else 'One-Time Buyer')

    if 'FirstName' in df.columns and 'LastName' in df.columns:
        df['CustomerName'] = df['FirstName'] + ' ' + df['LastName']
    elif 'FirstName' in df.columns:
        df['CustomerName'] = df['FirstName']

    if 'CustomerName' in df.columns:
        names = df[['CustomerID', 'CustomerName']].drop_duplicates()
        behavior_table = behavior_table.merge(names, on='CustomerID', how='left')
    else:
        behavior_table['CustomerName'] = 'N/A'


    clustering_features = behavior_table[['TotalOrders']].fillna(0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(clustering_features)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    behavior_table['Cluster'] = kmeans.fit_predict(scaled)

    if selected_segment != 'All':
        behavior_table = behavior_table[behavior_table['BehaviorType'] == selected_segment]

    csv_path = os.path.join(app.static_folder, 'behavior_output.csv')
    behavior_table.to_csv(csv_path, index=False)
    download_link = url_for('static', filename='behavior_output.csv')

    table = behavior_table.to_html(classes='table table-bordered table-striped table-hover display', index=False, table_id="behaviorTable")
    table += '<script src="https://cdn.datatables.net/1.10.24/js/jquery.dataTables.min.js"></script><script>$(document).ready(function() { $("#behaviorTable").DataTable({ pageLength: 10 }); });</script>'

    behavior_counts = behavior_table['BehaviorType'].value_counts()
    pie_data = {'data': [{'labels': behavior_counts.index.tolist(), 'values': behavior_counts.values.tolist(), 'type': 'pie'}], 'layout': {'title': 'Customer Behavior Types'}}
    pie_chart = f'<div id="behavior-pie" style="width:100%; height:400px;"></div><script src="https://cdn.plot.ly/plotly-latest.min.js"></script><script>Plotly.newPlot("behavior-pie", {json.dumps(pie_data)});</script>'

    avg_order_counts = behavior_table.groupby('BehaviorType')['TotalOrders'].mean().sort_values(ascending=False)
    bar_data = {'data': [{'x': avg_order_counts.index.tolist(), 'y': avg_order_counts.values.tolist(), 'type': 'bar', 'text': [f"{x:.2f}" for x in avg_order_counts.values], 'textposition': 'auto'}], 'layout': {'title': 'Avg Order Count per Behavior Type', 'yaxis': {'title': 'Average Orders'}}}
    bar_chart = f'<div id="behavior-bar" style="width:100%; height:400px;"></div><script>Plotly.newPlot("behavior-bar", {json.dumps(bar_data)});</script>'

    monthly_orders = df.groupby('OrderMonth')['CustomerID'].count()
    line_data = {'data': [{'x': monthly_orders.index.tolist(), 'y': monthly_orders.values.tolist(), 'mode': 'lines+markers', 'type': 'scatter', 'line': {'shape': 'linear'}}], 'layout': {'title': 'Total Orders Over Time', 'xaxis': {'title': 'Month'}, 'yaxis': {'title': 'Number of Orders'}}}
    line_chart = f'<div id="behavior-line" style="width:100%; height:400px;"></div><script>Plotly.newPlot("behavior-line", {json.dumps(line_data)});</script>'

    segment_summary = {"Total Customers": behavior_table.shape[0], "Most Common Behavior Type": behavior_counts.idxmax(), "Most Common Count": behavior_counts.max(), "Unique Behavior Types": len(behavior_counts), "Average Orders per Customer": round(behavior_table['TotalOrders'].mean(), 2)}

    return render_template('behavior.html', error=error, pie_chart=pie_chart, bar_chart=bar_chart, line_chart=line_chart, table=table, download_link=download_link, segment_summary=segment_summary, selected_segment=selected_segment)


@app.route('/product_preference')
def product_preference_analysis():
    error = None
    pie_chart = None
    bar_chart = None
    line_chart = None
    pref_results = None
    segment_summary = None

    if 'last_uploaded_file_id' in session and session['last_uploaded_file_id'] in uploaded_data:
        df = uploaded_data[session['last_uploaded_file_id']]
        if 'Spending_Score' in df.columns and 'ProductCategory' in df.columns:
            category_counts = df['ProductCategory'].value_counts()
            pie_data = {'data': [{'labels': category_counts.index.tolist(), 'values': category_counts.values.tolist(), 'type': 'pie'}], 'layout': {'title': 'Product Category Distribution'}}
            pie_chart = f'<div id="pie-chart" style="width:100%; height:400px;"></div><script src="https://cdn.plot.ly/plotly-latest.min.js"></script><script>Plotly.newPlot("pie-chart", {json.dumps(pie_data)});</script>'

            avg_scores = df.groupby('ProductCategory')['Spending_Score'].mean().sort_values(ascending=False)
            bar_data = {'data': [{'x': avg_scores.index.tolist(), 'y': avg_scores.values.tolist(), 'type': 'bar', 'text': [f"{val:.2f}" for val in avg_scores.values], 'textposition': 'auto'}], 'layout': {'title': 'Average Spending Score by Product Category'}}
            bar_chart = f'<div id="bar-chart" style="width:100%; height:400px;"></div><script>Plotly.newPlot("bar-chart", {json.dumps(bar_data)});</script>'

            if 'OrderDate' in df.columns:
                df['OrderDate'] = pd.to_datetime(df['OrderDate'])
                df['OrderMonth'] = df['OrderDate'].dt.to_period('M').astype(str)
                monthly_orders = df.groupby('OrderMonth')['CustomerID'].count()
                line_data = {'data': [{'x': monthly_orders.index.tolist(), 'y': monthly_orders.values.tolist(), 'mode': 'lines+markers', 'type': 'scatter'}], 'layout': {'title': 'Monthly Product Orders Over Time'}}
                line_chart = f'<div id="line-chart" style="width:100%; height:400px;"></div><script>Plotly.newPlot("line-chart", {json.dumps(line_data)});</script>'

            pref_results = df.groupby('ProductCategory').agg({'CustomerID': 'count', 'Spending_Score': 'mean'}).reset_index()
            pref_results.columns = ['Product Category', 'Customer Count', 'Average Spending Score']
            pref_results['Average Spending Score'] = pref_results['Average Spending Score'].round(2)

            segment_summary = {'Total Categories': len(category_counts), 'Most Popular Category': category_counts.idxmax(), 'Total Customers': df['CustomerID'].nunique(), 'Avg Spending Score': round(df['Spending_Score'].mean(), 2)}
        else:
            error = "Uploaded CSV must contain Spending_Score and ProductCategory columns."
    else:
        error = "No data available. Please upload a valid CSV."

    return render_template('product_preference.html', error=error, pie_chart=pie_chart, bar_chart=bar_chart, line_chart=line_chart, pref_results=pref_results, segment_summary=segment_summary)


@app.route('/geo')
def geo_analysis():
    error = None
    geo_pie_chart = None
    geo_bar_chart = None
    geo_line_chart = None
    geo_summary = None
    geo_results = None

    if 'last_uploaded_file_id' in session and session['last_uploaded_file_id'] in uploaded_data:
        df = uploaded_data[session['last_uploaded_file_id']]
        if 'Spending_Score' in df.columns and 'Region' in df.columns:
            df['Region'] = df['Region'].astype(str)
            geo_results = df.groupby('Region').agg({'CustomerID': 'count', 'Spending_Score': 'mean'}).reset_index()
            geo_results.columns = ['Region', 'Customer Count', 'Average Spending Score']
            geo_results['Average Spending Score'] = geo_results['Average Spending Score'].round(2)

            pie_data = {'data': [{'labels': geo_results['Region'].tolist(), 'values': geo_results['Customer Count'].tolist(), 'type': 'pie'}], 'layout': {'title': 'Customer Distribution by Region'}}
            geo_pie_chart = f'<div id="geo-pie"></div><script src="https://cdn.plot.ly/plotly-latest.min.js"></script><script>Plotly.newPlot("geo-pie", {json.dumps(pie_data)});</script>'

            bar_data = {'data': [{'x': geo_results['Region'].tolist(), 'y': geo_results['Average Spending Score'].tolist(), 'type': 'bar', 'marker': {'color': 'orange'}}], 'layout': {'title': 'Average Spending Score by Region', 'yaxis': {'title': 'Avg Spending'}}}
            geo_bar_chart = f'<div id="geo-bar"></div><script>Plotly.newPlot("geo-bar", {json.dumps(bar_data)});</script>'

            if 'OrderDate' in df.columns:
                df['OrderDate'] = pd.to_datetime(df['OrderDate'])
                df['Month'] = df['OrderDate'].dt.to_period('M').astype(str)
                monthly = df.groupby('Month')['Spending_Score'].mean().reset_index()
                line_data = {'data': [{'x': monthly['Month'].tolist(), 'y': monthly['Spending_Score'].round(2).tolist(), 'mode': 'lines+markers', 'line': {'color': 'blue'}}], 'layout': {'title': 'Average Spending Over Time', 'xaxis': {'title': 'Month'}, 'yaxis': {'title': 'Avg Spending'}}}
                geo_line_chart = f'<div id="geo-line"></div><script>Plotly.newPlot("geo-line", {json.dumps(line_data)});</script>'

            geo_summary = {"Total Regions": df['Region'].nunique(), "Total Customers": df['CustomerID'].nunique(), "Overall Avg Spending Score": f"{df['Spending_Score'].mean():.2f}", "Max Region Spending Avg": f"{geo_results['Average Spending Score'].max():.2f}", "Min Region Spending Avg": f"{geo_results['Average Spending Score'].min():.2f}"}
        else:
            error = "Uploaded CSV must contain 'Spending_Score' and 'Region' columns."
    else:
        error = "No data available. Please upload a valid CSV."

    return render_template('geo.html', error=error, geo_pie_chart=geo_pie_chart, geo_bar_chart=geo_bar_chart, geo_line_chart=geo_line_chart, geo_summary=geo_summary, geo_results=geo_results)


@app.route('/trends')
def trends_analysis():
    error = None
    line_chart = None
    bar_chart = None
    pie_chart = None
    trends_results = pd.DataFrame()
    download_link = None
    selected_month = request.args.get('month', 'All')

    file_id = session.get('last_uploaded_file_id')
    if file_id and file_id in uploaded_data:
        df = uploaded_data[file_id]
        if 'Spending_Score' in df.columns and ('Date' in df.columns or 'PurchaseDate' in df.columns):
            time_column = 'Date' if 'Date' in df.columns else 'PurchaseDate'
            df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
            df.dropna(subset=[time_column], inplace=True)
            df['Month'] = df[time_column].dt.to_period('M').astype(str)

            if selected_month != 'All':
                df = df[df['Month'] == selected_month]

            trends_results = df.groupby('Month').agg({'CustomerID': 'count', 'Spending_Score': 'mean'}).reset_index()
            trends_results.columns = ['Month', 'Customer Count', 'Average Spending Score']
            trends_results['Average Spending Score'] = trends_results['Average Spending Score'].round(2)

            pie_data = {'data': [{'labels': trends_results['Month'].tolist(), 'values': trends_results['Customer Count'].tolist(), 'type': 'pie'}], 'layout': {'title': 'Customer Purchase Share by Month'}}
            pie_chart = f'<div id="pie-chart"></div><script src="https://cdn.plot.ly/plotly-latest.min.js"></script><script>Plotly.newPlot("pie-chart", {json.dumps(pie_data)});</script>'

            line_data = {'data': [{'x': trends_results['Month'].tolist(), 'y': trends_results['Average Spending Score'].tolist(), 'type': 'scatter', 'mode': 'lines+markers', 'name': 'Avg Spending'}], 'layout': {'title': 'Purchase Trends Over Time', 'xaxis': {'title': 'Month'}, 'yaxis': {'title': 'Avg Spending Score'}}}
            line_chart = f'<div id="line-chart"></div><script>Plotly.newPlot("line-chart", {json.dumps(line_data)});</script>'

            bar_data = {'data': [{'x': trends_results['Month'].tolist(), 'y': trends_results['Customer Count'].tolist(), 'type': 'bar', 'name': 'Customer Count'}], 'layout': {'title': 'Customer Count by Month', 'xaxis': {'title': 'Month'}, 'yaxis': {'title': 'Count'}}}
            bar_chart = f'<div id="bar-chart"></div><script>Plotly.newPlot("bar-chart", {json.dumps(bar_data)});</script>'

            output_file = f"trend_results_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_file)
            trends_results.to_csv(output_path, index=False)
            download_link = url_for('download_trends', filename=output_file)
        else:
            error = "CSV must contain 'Spending_Score' and either 'Date' or 'PurchaseDate'."
    else:
        error = "No data available. Please upload a CSV file first."

    return render_template('trends.html', error=error, line_chart=line_chart, bar_chart=bar_chart, pie_chart=pie_chart, trends_results=trends_results, download_link=download_link, selected_month=selected_month)


@app.route('/predictive')
def predictive_analysis():
    error = None
    line_chart = None
    bar_chart = None
    pie_chart = None
    predictive_analysis_results = pd.DataFrame()
    download_link = None
    month_list = []
    selected_month = request.args.get('month', 'All')

    if 'last_uploaded_file_id' in session and session['last_uploaded_file_id'] in uploaded_data:
        df = uploaded_data[session['last_uploaded_file_id']]
        if 'Spending_Score' in df.columns and ('Date' in df.columns or 'PurchaseDate' in df.columns):
            time_column = 'Date' if 'Date' in df.columns else 'PurchaseDate'
            df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
            df.dropna(subset=[time_column], inplace=True)
            df['Month'] = df[time_column].dt.to_period('M').astype(str)
            month_list = df['Month'].unique().tolist()

            if selected_month != 'All':
                df = df[df['Month'] == selected_month]

            monthly_avg = df.groupby('Month')['Spending_Score'].mean().sort_index().reset_index()
            if not monthly_avg.empty:
                last_month = monthly_avg['Month'].iloc[-1]
                next_month = (pd.Period(last_month, freq='M') + 1).strftime('%Y-%m')
                predicted_value = monthly_avg['Spending_Score'].iloc[-1] * 1.1

                predictive_analysis_results = pd.DataFrame({'Month': [next_month], 'Predicted Average Spending Score': [round(predicted_value, 2)]})

                months = monthly_avg['Month'].tolist() + [next_month]
                scores = monthly_avg['Spending_Score'].tolist() + [predicted_value]

                pie_data = {'data': [{'labels': monthly_avg['Month'].tolist(), 'values': monthly_avg['Spending_Score'].tolist(), 'type': 'pie'}], 'layout': {'title': 'Spending Share by Month'}}
                pie_chart = f'<div id="pie-chart"></div><script src="https://cdn.plot.ly/plotly-latest.min.js"></script><script>Plotly.newPlot("pie-chart", {json.dumps(pie_data)});</script>'

                line_data = {'data': [{'x': months, 'y': scores, 'type': 'scatter', 'mode': 'lines+markers', 'name': 'Predicted Spending'}], 'layout': {'title': 'Predicted Trends Over Time', 'xaxis': {'title': 'Month'}, 'yaxis': {'title': 'Avg Spending Score'}}}
                line_chart = f'<div id="line-chart"></div><script src="https://cdn.plot.ly/plotly-latest.min.js"></script><script>Plotly.newPlot("line-chart", {json.dumps(line_data)});</script>'

                bar_data = {'data': [{'x': months, 'y': scores, 'type': 'bar', 'name': 'Predicted Spending'}], 'layout': {'title': 'Predicted Monetary Value by Period', 'xaxis': {'title': 'Month'}, 'yaxis': {'title': 'Avg Spending Score'}}}
                bar_chart = f'<div id="bar-chart"></div><script src="https://cdn.plot.ly/plotly-latest.min.js"></script><script>Plotly.newPlot("bar-chart", {json.dumps(bar_data)});</script>'

                output_file = f"predictive_results_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
                path = os.path.join(app.config['OUTPUT_FOLDER'], output_file)
                predictive_analysis_results.to_csv(path, index=False)
                download_link = url_for('download_predictive', filename=output_file)
        else:
            error = "Uploaded CSV must contain 'Spending_Score' and either 'Date' or 'PurchaseDate'."
    else:
        error = "No data available. Please upload a valid CSV."

    return render_template('predictive_analysis.html', error=error, line_chart=line_chart, bar_chart=bar_chart, pie_chart=pie_chart, predictive_analysis_results=predictive_analysis_results, download_link=download_link, selected_month=selected_month, month_list=month_list)


@app.route('/results')
def results():
    error = None
    detailed_results = None
    results_summary = None
    combined_charts = {}
    output_file = None
    selected_month = request.args.get('month', 'All')
    month_list = []

    try:
        if 'last_uploaded_file_id' in session and session['last_uploaded_file_id'] in uploaded_data:
            df = uploaded_data[session['last_uploaded_file_id']]

            time_column = None
            if 'Date' in df.columns:
                time_column = 'Date'
            elif 'PurchaseDate' in df.columns:
                time_column = 'PurchaseDate'

            if time_column:
                df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
                df.dropna(subset=[time_column], inplace=True)
                df['Month'] = df[time_column].dt.to_period('M').astype(str)
                month_list = df['Month'].unique().tolist()
                if selected_month != 'All':
                    df = df[df['Month'] == selected_month]
            else:
                error = "Missing 'Date' or 'PurchaseDate' column for month filtering."

            if 'CustomerID' in df.columns:
                if 'Spending_Score' in df.columns and 'Annual_Income' in df.columns:
                    detailed_results = df[['CustomerID', 'Spending_Score', 'Annual_Income']]
                else:
                    detailed_results = df[['CustomerID']]

                total_customers = len(df)
                avg_spending = df['Spending_Score'].mean() if 'Spending_Score' in df.columns else 0
                avg_income = df['Annual_Income'].mean() if 'Annual_Income' in df.columns else 0
                results_summary = {"total_customers": total_customers, "avg_spending": f"{avg_spending:.2f}", "avg_income": f"{avg_income:.2f}"}

                if 'Segment' in df.columns:
                    pie_fig = px.pie(df, names='Segment', title='Customer Segment Distribution')
                    combined_charts['Customer Segment Distribution'] = to_html(pie_fig, full_html=False)

                if time_column and 'Spending_Score' in df.columns:
                    df_sorted = df.sort_values(time_column)
                    if not df_sorted[time_column].isnull().all():
                        line_fig = px.line(df_sorted, x=time_column, y='Spending_Score', title='Spending Score Over Time')
                        combined_charts['Spending Score Over Time'] = to_html(line_fig, full_html=False)

                if 'Annual_Income' in df.columns and 'Segment' in df.columns:
                    bar_df = df.groupby('Segment')['Annual_Income'].mean().reset_index()
                    bar_fig = px.bar(bar_df, x='Segment', y='Annual_Income', title='Average Income by Segment')
                    combined_charts['Average Income by Segment'] = to_html(bar_fig, full_html=False)
            else:
                error = "Uploaded CSV must contain a 'CustomerID' column."
        else:
            error = "No data available. Please upload a valid CSV first."
    except Exception as e:
        error = f"An error occurred during processing: {str(e)}"

    return render_template('results.html', error=error, results_summary=results_summary, detailed_results=detailed_results, combined_charts=combined_charts, output_file=output_file, selected_month=selected_month, month_list=month_list)


# ─────────────────────────────────────────────
# DEMOGRAPHIC ROUTE
# ─────────────────────────────────────────────

@app.route('/demographic_trends', endpoint='demographic_trends_analysis')
@login_required
def demographic_trends():
    error = None
    visualizations = []
    demographic_trends_results = None
    download_link = None
    selected_month = request.args.get('month', 'All')
    month_list = []

    if 'last_uploaded_file_id' in session and session['last_uploaded_file_id'] in uploaded_data:
        df = uploaded_data[session['last_uploaded_file_id']]

        if 'OrderDate' not in df.columns:
            if 'PurchaseDate' in df.columns:
                df.rename(columns={'PurchaseDate': 'OrderDate'}, inplace=True)
            else:
                error = "CSV must contain 'OrderDate' or 'PurchaseDate' for time filtering."
                return render_template('demographic.html', error=error, visualizations=[], demo_results=None, selected_month=selected_month, month_list=[], download_link=None)

        df['OrderDate'] = pd.to_datetime(df['OrderDate'], errors='coerce')
        df.dropna(subset=['OrderDate'], inplace=True)
        df['Month'] = df['OrderDate'].dt.to_period('M').astype(str)
        month_list = sorted(df['Month'].unique().tolist())

        if selected_month != 'All':
            df = df[df['Month'] == selected_month]

        from demographic import calculate_demographic_trends
        trends, output_file, visualizations, error = calculate_demographic_trends(df)

        if trends and not error:
            numeric_trends = {k: pd.Series(v['trend']) for k, v in trends.items() if v['type'] == 'numeric'}
            combined_df = pd.DataFrame(numeric_trends).reset_index()
            demographic_trends_results = combined_df.to_html(classes='table table-bordered table-striped display', index=False, table_id='demographicTrendsTable')
            filename = os.path.basename(output_file)
            download_link = url_for('download_demographic', filename=filename)
        else:
            demographic_trends_results = None
    else:
        error = "No uploaded CSV found. Please upload a file first."

    return render_template('demographic.html', error=error, visualizations=visualizations, demo_results=demographic_trends_results, selected_month=selected_month, month_list=month_list, download_link=download_link)


# ─────────────────────────────────────────────
# CEO ROUTES
# ─────────────────────────────────────────────

def load_uploaded_data():
    try:
        file_id = session.get('last_uploaded_file_id')
        if file_id and file_id in uploaded_data:
            return uploaded_data[file_id]
    except Exception:
        pass
    return pd.DataFrame()


@app.route('/ceo/business_strategies')
@login_required
@role_required('CEO')
def ceo_business_strategies():
    data = load_uploaded_data()
    if data.empty:
        return render_template('error.html', error='No data', message='No uploaded dataset found. Please upload a CSV first.')

    required = {'SalesAmount', 'CustomerID'}
    if not required.issubset(set(data.columns)):
        return render_template('error.html', error='Missing columns', message=f'Required columns missing: {required - set(data.columns)}')

    revenue = data['SalesAmount'].sum()
    orders = len(data)
    customers = data['CustomerID'].nunique()
    insights = {'revenue': revenue, 'orders': orders, 'customers': customers}

    strategy = f"""
    Based on current business performance:
    - Total revenue: ${revenue:,.2f}
    - Total customers: {customers}
    - Total orders: {orders}

    ✅ AI-Driven Strategies:
    1. Launch a targeted loyalty and rewards program to strengthen repeat purchase behavior and boost customer lifetime value.
    2. Deploy personalized retention offers to proactively reduce churn in at-risk segments.
    3. Scale investment in your highest-performing product categories.
    4. Prioritize expansion in high-growth regions where demand velocity is outpacing the average market trend.
    5. Apply dynamic pricing and discount optimization to improve turnover for slow-moving categories.
    """

    return render_template('ceo_business_strategies.html', insights=insights, strategy=strategy)


@app.route('/ceo/financial_trends')
@login_required
@role_required('CEO')
def ceo_financial_trends():
    data = load_uploaded_data()
    if data.empty:
        return render_template('error.html', error='No data', message='No uploaded dataset found.')

    if 'PurchaseDate' not in data.columns or 'SalesAmount' not in data.columns:
        return render_template('error.html', error='Missing columns', message="Required columns 'PurchaseDate' or 'SalesAmount' missing.")

    data = data.copy()
    data['PurchaseDate'] = pd.to_datetime(data['PurchaseDate'], errors='coerce')
    data.dropna(subset=['PurchaseDate'], inplace=True)

    monthly = data.groupby(data['PurchaseDate'].dt.to_period('M')).agg({'SalesAmount': 'sum'}).reset_index()
    monthly['PurchaseDate'] = monthly['PurchaseDate'].astype(str)

    fig = px.line(monthly, x='PurchaseDate', y='SalesAmount', title='Monthly Revenue Trend')
    chart_html = fig.to_html(full_html=False)

    return render_template('ceo_financial_trends.html', chart_html=chart_html)


@app.route('/ceo/performance_metrics')
@login_required
@role_required('CEO')
def ceo_performance_metrics():
    data = load_uploaded_data()
    if data.empty:
        return render_template('error.html', error='No data', message='No uploaded dataset found.')

    required = {'SalesAmount', 'CustomerID', 'ProductCategory'}
    if not required.issubset(set(data.columns)):
        missing = required - set(data.columns)
        return render_template('error.html', error='Missing columns', message=f'Required columns missing: {missing}')

    metrics = {
        'avg_order_value': float(data['SalesAmount'].mean()),
        'purchase_frequency': float(len(data) / data['CustomerID'].nunique()) if data['CustomerID'].nunique() > 0 else 0,
        'customer_retention': float(data['CustomerID'].value_counts().mean()),
        'top_product': data['ProductCategory'].value_counts().idxmax() if 'ProductCategory' in data.columns and not data['ProductCategory'].empty else None
    }

    return render_template('ceo_performance_metrics.html', metrics=metrics)


# ─────────────────────────────────────────────
# DOWNLOAD ROUTES
# ─────────────────────────────────────────────

@app.route('/download_demographic/<filename>')
def download_demographic(filename):
    path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return f"File {filename} not found.", 404


@app.route('/download_trends/<filename>')
def download_trends(filename):
    path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return f"File {filename} not found.", 404


@app.route('/download_predictive/<filename>')
def download_predictive(filename):
    path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return f"File {filename} not found.", 404


@app.route('/download_results/<filename>')
def download_results(filename):
    try:
        return send_from_directory(directory=app.config['OUTPUT_FOLDER'], path=filename, as_attachment=True)
    except FileNotFoundError:
        return "File not found", 404


if __name__ == '__main__':
    app.run(debug=True)
