from flask import Flask, render_template, jsonify, request, Response, stream_with_context
import pandas as pd 
import joblib
import requests
import markdown2
import json

app= Flask(__name__)

PORT_CSV_FILE= "response_portcall_269_simple.csv"
BERTH_CSV_FILE= "response_berthcall_269_kandla.csv"

def load_port_data():
    df=pd.read_csv(PORT_CSV_FILE, on_bad_lines='skip')
    df.columns=df.columns.str.strip() # Removing whitespaces from the column names
    df=df[df['PORT_NAME'].str.strip().str.upper()=='KANDLA']
    df['TIMESTAMP_UTC']=pd.to_datetime(df['TIMESTAMP_UTC'])
    df['Month']=df['TIMESTAMP_UTC'].dt.to_period('M')
    df['Year']=df['TIMESTAMP_UTC'].dt.to_period('Y')
    df['MARKET'] = df['MARKET'].astype(str).str.strip() # If market column data has inconsistent spaces then strip spaces
    return df

def load_berth_data(include_kan3=True):
    df=pd.read_csv(BERTH_CSV_FILE, on_bad_lines='skip')
    df.columns=df.columns.str.strip()
    df=df[df['PORT_NAME'].str.strip().str.upper()=='KANDLA']

    
    df['BERTH_NAME']=df['BERTH_NAME'].astype(str).str.strip()
    
    if not include_kan3:
        df=df[df['BERTH_NAME']!='KAN3']

    # Parse both timestamps
    if 'DOCK_TIMESTAMP_UTC' in df.columns:
        df['DOCK_TIMESTAMP_UTC'] = pd.to_datetime(df['DOCK_TIMESTAMP_UTC'], errors='coerce')
    if 'UNDOCK_TIMESTAMP_UTC' in df.columns:
        df['UNDOCK_TIMESTAMP_UTC'] = pd.to_datetime(df['UNDOCK_TIMESTAMP_UTC'], errors='coerce')
    
    return df

@app.route('/')
def index():
    return render_template('index2.html')



# ************************************************* PORT-CHART APIs START*************************************************

# API endpoint for summary statistics like total ships count in current year, montly average, active ships count and port utilization
@app.route('/api/summary_stats')
def summary_stats():
    df=load_port_data()
    # current_year = pd.Timestamp.now().year
    current_year=2024
    df['TIMESTAMP_UTC'] = pd.to_datetime(df['TIMESTAMP_UTC'], errors='coerce')
    df_current_year = df[df['TIMESTAMP_UTC'].dt.year == current_year]

    total_ships_current_year = df_current_year['SHIP_ID'].count()
    # total_ships_current_year = len(df_current_year)
    # Monthly average = total / number of unique months in year
    months_in_year = df_current_year['TIMESTAMP_UTC'].dt.month.nunique()  

    monthly_avg = round(total_ships_current_year / months_in_year, 0) if months_in_year > 0 else 0
    # Get the last date (not timestamp) in the data
    max_date = df['TIMESTAMP_UTC'].dt.date.max()
    print("Max date in data:")
    print(max_date)
    # Filter rows with the last date
    df_last_day = df[df['TIMESTAMP_UTC'].dt.date == max_date]
    # Count all ship entries (including duplicates)
    # active_ships_today = len(df_last_day)
    active_ships_today = df_last_day['SHIP_ID'].count()
    # Port Utilization = (active / total in day) * 100
    utilization = round((active_ships_today / 72) * 100, 2) if total_ships_current_year > 0 else 0
    return jsonify({
        "total_ships": int(total_ships_current_year),
        "monthly_avg": float(monthly_avg),
        "active_ships": int(active_ships_today),
        "port_utilization": f"{utilization}%"  # Or just return the number if you want formatting on frontend
    })


# API endpoint for monthly ship count and forecast
@app.route('/api/monthly_ship_count_and_forecast')
def monthly_ship_count_and_forecast():
    
    # Load original data
    df = pd.read_csv("/home/dhanjay/Desktop/Kandla_Port_Dashboard/monthly_ship_counts_(2016-2025).csv", on_bad_lines='skip')
    df.rename(columns={'Date': 'ds', 'Ship_Count': 'y'}, inplace=True)
    df['ds'] = pd.to_datetime(df['ds'], format='%Y-%m')
    df=df[df['ds']>=pd.Timestamp('2018-01-01')]

    # Historical data
    historical_labels = df['ds'].dt.strftime('%Y-%m').tolist()
    historical_values = df['y'].astype(int).tolist()

    # Load saved Prophet model
    model = joblib.load('prophet_model_ship_monthlyCnt.pkl')

    # Forecast 6 months ahead
    future = model.make_future_dataframe(periods=13, freq='M')
    forecast = model.predict(future)

    # Filter future only
    # future_forecast = forecast[forecast['ds'] > df['ds'].max()]
     # Convert to 'YYYY-MM' for precise filtering
    last_month_str = df['ds'].max().strftime('%Y-%m')
    forecast['month_str'] = forecast['ds'].dt.strftime('%Y-%m')
    # Filter only unseen future months
    future_forecast = forecast[forecast['month_str'] > last_month_str]

    predicted_labels = future_forecast['ds'].dt.strftime('%Y-%m').tolist()
    predicted_values = future_forecast['yhat'].round(0).astype(int).tolist()


    return jsonify({
        "historical": {
            "labels": historical_labels,
            "values": historical_values
        },
        "predicted": {
            "labels": predicted_labels,
            "values": predicted_values
        }
    })


# API endpoint to get top 5 ship types
@app.route('/api/top_ship_types')
def top_ship_types():
    df=load_port_data()
    top_types=df['TYPE_NAME'].value_counts().head(5)
    return jsonify({
        "labels": [str(label) for label in top_types.index], # Ship types in string format
        "values": [int(value) for value in top_types.values]  # Count of ships in integer format for each type
    })


# API endpoint to get top 10 frequent ships
@app.route('/api/top_frequent_ships')
def top_frequent_ships():
    df=load_port_data()
    top_ships=df['SHIPNAME'].value_counts().head(10)
    return jsonify({
        "labels": [str(label) for label in top_ships.index],  # Ship names in string format
        "values": [int(value) for value in top_ships.values]  # Count of ships in integer format
    })


# API endpoint to get market distribution
@app.route('/api/market_distribution')
def market_distribution():
    df = load_port_data()
    df['MARKET'] = df['MARKET'].astype(str).str.strip()  # Cleanup
    
    market_counts = df['MARKET'].dropna().value_counts()

    if market_counts.empty:
        return jsonify({"labels": [], "values": []})

    top_5 = market_counts.head(5)
    remaining = market_counts.iloc[5:].sum()

    labels = top_5.index.tolist()
    values = top_5.tolist()

    if remaining > 0:
        labels.append("REMAINING MARKETS")
        values.append(remaining)

    # Normalize to make sure the total = 100%
    total = sum(values)
    values = [round((v / total) * 100, 1) for v in values]

    return jsonify({
        "labels": labels,
        "values": values
    })

# ************************************************* PORT-CHART APIs END*************************************************


# ************************************************* BERTH-CHART APIs START*************************************************

# API for calculating berth usage of top 10 berths of Kandla Port
@app.route('/api/berth_utilization')
def berth_utilization():
    include_kan3=request.args.get('includeKAN3','yes')=='yes'
    df=load_berth_data(include_kan3=include_kan3)
    # df_2024= df[df['DOCK_TIMESTAMP_UTC'].dt.year == 2024]
    total_ships_2024=df['SHIP_ID'].count()
    berth_counts = df.groupby('BERTH_NAME')['SHIP_ID'].count()
    top_10_berths = berth_counts.sort_values(ascending=False)
    berth_utilization = (top_10_berths / total_ships_2024 * 100).round(2).head(10)
    result = berth_utilization.reset_index()
    result.columns = ['BERTH_NAME', 'Utilization (%)']
    return jsonify(result.to_dict(orient='records'))

# API endpoint for total berth usage of top 10 berths at Kandla Port
@app.route('/api/total_berth_usage')
def total_berth_usage():
    include_kan3 = request.args.get('includeKAN3', 'yes') == 'yes'
    df = load_berth_data(include_kan3=include_kan3) 
    df.columns = df.columns.str.strip() 
    
    # Filter rows for Kandla port only (case insensitive, strip spaces)
    df = df[df['PORT_NAME'].astype(str).str.strip().str.upper() == 'KANDLA']
    
    # Clean BERTH_NAME
    df['BERTH_NAME'] = df['BERTH_NAME'].astype(str).str.strip()
    
    berth_counts = df['BERTH_NAME'].value_counts().sort_values(ascending=False).head(10)

    return jsonify({
        "labels": berth_counts.index.tolist(),
        "values": berth_counts.values.tolist()
    })




# API endpoint for monthly berth usage trend
@app.route('/api/monthly_berth_usage')
def monthly_berth_usage():
    include_kan3=request.args.get('includeKAN3','yes')=='yes'
    df=load_berth_data(include_kan3=include_kan3)
    df.columns = df.columns.str.strip() 
    
    # Ensure DOCK_TIMESTAMP_UTC is datetime
    df['DOCK_TIMESTAMP_UTC'] = pd.to_datetime(df['DOCK_TIMESTAMP_UTC'], errors='coerce')
    
    # Create a Month column based on DOCK time
    df['DOCK_MONTH'] = df['DOCK_TIMESTAMP_UTC'].dt.to_period('M').astype(str)
    
    # Clean BERTH_NAME
    df['BERTH_NAME'] = df['BERTH_NAME'].astype(str).str.strip()

    # Group by Month and Berth
    grouped = df.groupby(['DOCK_MONTH', 'BERTH_NAME'])['SHIP_ID'].count().unstack().fillna(0)

    # Find top 5 berths by total ship count
    top_10_berths = grouped.sum(axis=0).sort_values(ascending=False).index.tolist()

    # Filter the grouped DataFrame to include only top 5 berths
    grouped_top5 = grouped[top_10_berths]

    return jsonify({
        "months": list(grouped_top5.index),  # ["2024-01", "2024-02", ...]
        "berths": list(grouped_top5.columns),  # ["BERTH A", "BERTH B", ...]
        "values": grouped_top5.values.tolist()  # 2D array of counts per berth per month
    })

# Each berth average time for dock and undock
@app.route('/api/average_berth_duration')
def average_berth_duration():
    include_kan3=request.args.get('includeKAN3','yes')=='yes'
    df=load_berth_data(include_kan3=include_kan3)
    df.columns = df.columns.str.strip()

    # Clean and convert timestamps
    df['DOCK_TIMESTAMP_UTC'] = pd.to_datetime(df['DOCK_TIMESTAMP_UTC'], errors='coerce')
    df['UNDOCK_TIMESTAMP_UTC'] = pd.to_datetime(df['UNDOCK_TIMESTAMP_UTC'], errors='coerce')

    # Strip berth names
    df['BERTH_NAME'] = df['BERTH_NAME'].astype(str).str.strip()

    # Filter valid timestamps
    df = df[df['DOCK_TIMESTAMP_UTC'].notnull() & df['UNDOCK_TIMESTAMP_UTC'].notnull()]

    # Calculate duration in hours
    df['DURATION_HOURS'] = (df['UNDOCK_TIMESTAMP_UTC'] - df['DOCK_TIMESTAMP_UTC']).dt.total_seconds() / 3600

    # Group by berth and calculate average duration
    avg_duration = df.groupby('BERTH_NAME')['DURATION_HOURS'].mean().sort_values(ascending=False)
    # print(avg_duration)
    return jsonify({
        "labels": avg_duration.index.tolist(),   # ["BERTH A", "BERTH B", ...]
        "values": avg_duration.round(2).tolist() # [12.5, 10.2, ...]
    })

@app.route('/api/berth_downtime')
def berth_downtime():
    include_kan3=request.args.get('includeKAN3','yes')=='yes'
    df=load_berth_data(include_kan3=include_kan3)

    # Convert timestamps
    df['DOCK_TIMESTAMP_UTC'] = pd.to_datetime(df['DOCK_TIMESTAMP_UTC'], errors='coerce')
    df['BERTH_NAME'] = df['BERTH_NAME'].astype(str).str.strip()

    max_date = df['DOCK_TIMESTAMP_UTC'].max().date()

    last_dock = df.groupby('BERTH_NAME')['DOCK_TIMESTAMP_UTC'].max().dt.date
    downtime_days = (max_date - last_dock).apply(lambda x: x.days)

    result = downtime_days[downtime_days > 2]  # Only show idle > 30 days
    result = result.sort_values(ascending=False).reset_index()
    result.columns = ['BERTH_NAME', 'Days_Since_Last_Use']

    return jsonify(result.to_dict(orient='records'))


# API end point to get each berth top ship types
@app.route('/api/all_berths')
def get_all_berths():
    include_kan3=request.args.get('includeKAN3','yes')=='yes'
    df=load_berth_data(include_kan3=include_kan3)
    berth_list = df['BERTH_NAME'].dropna().unique().tolist()
    return jsonify(sorted(berth_list))

@app.route('/api/top_3_ship_types_by_berth')
def top_3_ship_types_by_berth():
    berth_name = request.args.get('berth')
    if not berth_name:
        return jsonify({'error': 'berth parameter is required'}), 400

    include_kan3=request.args.get('includeKAN3','yes')=='yes'
    df=load_berth_data(include_kan3=include_kan3)
    df_filtered = df[df['BERTH_NAME'] == berth_name]

    top_ship_types = (
        df_filtered.groupby('TYPE_NAME')['SHIP_ID']
        .count()
        .reset_index(name='Ship_Count')
        .sort_values('Ship_Count', ascending=False)
        .head(3)
    )

    return jsonify(top_ship_types.to_dict(orient='records'))



# ************************************************* BERTH-CHART APIs END *************************************************


#  ************************************************* OLAMA-LLAMA3.1:70b APIs START*************************************************

# # API endpoint to generate insights based on port or berth data
# @app.route('/api/generate_insight')
# def generate_insight():
#     section = request.args.get('section', 'port')  # default to port

#     try:
#         if section == 'berth':
#             # Generate berth insight
#             df = load_berth_data()
#             summary = {"total_berths_on_kandla_port": 24}
#             top_10_berths_kandla_port_as_per_utilization= requests.get('http://localhost:5001/api/berth_utilization').json()
#             top_10_berths_as_per_usage = requests.get('http://localhost:5001/api/total_berth_usage').json()
#             avg_berth_duration_for_dockUndock_inHrs = requests.get('http://localhost:5001/api/average_berth_duration').json()
#             berths_which_are_idle_from_2_plus_days = requests.get('http://localhost:5001/api/berth_downtime').json()
#             # For berth downtime
#             idle_labels = [item["BERTH_NAME"] for item in berths_which_are_idle_from_2_plus_days]
#             idle_values = [item["Days_Since_Last_Use"] for item in berths_which_are_idle_from_2_plus_days]
#             # For berth utilization
#             util_labels = [item["BERTH_NAME"] for item in top_10_berths_kandla_port_as_per_utilization]
#             util_values = [item["Utilization (%)"] for item in top_10_berths_kandla_port_as_per_utilization]


#             prompt = f"""
# You are a senior data analyst at Kandla Port. You have been given operational data related to port berths.

# Your task is to generate clear, helpful, and human-friendly insights from the following data. Focus on identifying **strengths**, **bottlenecks**, and **areas needing attention**, while also suggesting improvements.

# 📊 **Data Summary**:
# - Top 10 Berths by Utilization: {util_labels} with values {util_values}
# - Total Berths in Use: {summary['total_berths_on_kandla_port']}
# - Top 10 Berths by Usage: {top_10_berths_as_per_usage['labels']} with values {top_10_berths_as_per_usage['values']}
# - Average Docking & Undocking Duration (hours): {avg_berth_duration_for_dockUndock_inHrs['labels']} with durations {avg_berth_duration_for_dockUndock_inHrs['values']}
# - Berths Idle for 2+ days: {idle_labels} with values {idle_values}

# 📝 **Instructions**:
# - Present your insights as **bullet points** using circle bullets (•), each on a **new line** (`\n`).
# - Start each point with a **bold heading** (no stars), and add relevant **emojis** for visual cues.
# - For key words like “increase”, “decrease”, “delay”, “maintenance”, or “alert”, briefly **explain the cause or impact** before the keyword.
# - Use a **clear, simple tone** that even a non-technical stakeholder can understand.
# - Make sure to include:
#     - Berths performing **exceptionally well** 🚀
#     - Berths showing **delays or inefficiencies** 🐢
#     - Possible reasons for long turnaround time ⏱️
#     - Any outliers or inconsistencies worth noting 🔍

# 🔧 **Lastly**, suggest 3-4 **practical improvement ideas** to increase berth efficiency. Write them in bullet points using new lines (`\n`), based on the data.

# Output:
# """

#         else:
#             # Generate port-level insight
#             summary = requests.get('http://localhost:5001/api/summary_stats').json()
#             monthly_ship_cnt = requests.get('http://localhost:5001/api/monthly_ship_count_and_forecast').json()
#             top_5_ship_types = requests.get('http://localhost:5001/api/top_ship_types').json()
#             top_10_frequent_ships = requests.get('http://localhost:5001/api/top_frequent_ships').json()
#             market_distribution = requests.get('http://localhost:5001/api/market_distribution').json()

#             prompt = f"""
# You are a senior data analyst at Kandla Port. You are provided with key metrics and traffic data for the port.

# Your job is to extract **meaningful**, **easy-to-read** insights that highlight:
# - Current port performance 📈
# - Major trends 📊
# - Bottlenecks or alerts 🚨
# - Opportunities for improvement ⚙️

# 📊 **Port Data**:
# - Yearly Ship Count: {summary['total_ships']}
# - Monthly Average: {summary['monthly_avg']}
# - Active Ships Today: {summary['active_ships']}
# - Port Utilization: {summary['port_utilization']}%
# - Historical Ship Traffic (Jan 2018 – Apr 2025): {monthly_ship_cnt['historical']['values']}
# - Forecasted Ship Traffic (Next 1 year): {monthly_ship_cnt['predicted']['values']}
# - Top 5 Ship Types: {top_5_ship_types['labels']} with values {top_5_ship_types['values']}
# - Most Frequent Ships (Top 10): {top_10_frequent_ships['labels']} with values {top_10_frequent_ships['values']}
# - Market Distribution: {market_distribution['labels']} with values {market_distribution['values']}

# 📝 **Instructions**:
# - Format your insights as bullet points (•), each on a new line (`\n`).
# - Start each with a **bold** heading and relevant emojis.
# - Highlight any **alerts**, **rising trends**, or **sudden drops** with a short cause or note.
# - Keep the language **non-technical** and clear for senior decision-makers.
# - Cover:
#     - Port traffic trends 📉📈
#     - Overutilized or underused resources 📊
#     - Consistency or spikes in ship arrivals
#     - Any potential risk or delay markers 🛑

# 🔧 **Finish with 3–4 smart suggestions** to improve port throughput, traffic handling, or resource planning. Write them in bullet points on new lines (`\n`).

# Output:
# """

#         # Call LLM API
#         response = requests.post(
#             "http://192.168.10.41:11434/api/chat",
            
#             #   "http://localhost:11434/api/chat",

#             json={
#                 # "model": "gemma3:4b",
#                 "model": "llama3.3:70b",
#                 "messages": [{"role": "user", "content": prompt}],
#                 # "prompt": prompt,
#                 "stream": False
#             },
#             headers={"Content-Type": "application/json"}
#         )
#         response.raise_for_status()
#         result = response.json()
#         # content = result.get("message", {}).get("content", "No insight generated.")

#         # return jsonify({"insight": content})
#         markdown_text = result.get("message", {}).get("content", "No insight generated.")
#         html_content = markdown2.markdown(markdown_text)

#         return jsonify({"insight": html_content})

#     except Exception as e:
#         return jsonify({"insight": f"Error generating insight: {str(e)}"})


# # ************************************************* OLAMA-LLAMA3.1:70b APIs END *************************************************


# API endpoint to generate insights based on port or berth data
@app.route('/api/generate_insight')
def generate_insight():
    section = request.args.get('section', 'port')  # default to port

    try:
        if section == 'berth':
            # Generate berth insight
            df = load_berth_data()
            summary = {"total_berths_on_kandla_port": 24}
            top_10_berths_kandla_port_as_per_utilization= requests.get('http://localhost:5001/api/berth_utilization').json()
            top_10_berths_as_per_usage = requests.get('http://localhost:5001/api/total_berth_usage').json()
            avg_berth_duration_for_dockUndock_inHrs = requests.get('http://localhost:5001/api/average_berth_duration').json()
            berths_which_are_idle_from_2_plus_days = requests.get('http://localhost:5001/api/berth_downtime').json()
            # For berth downtime
            idle_labels = [item["BERTH_NAME"] for item in berths_which_are_idle_from_2_plus_days]
            idle_values = [item["Days_Since_Last_Use"] for item in berths_which_are_idle_from_2_plus_days]
            # For berth utilization
            util_labels = [item["BERTH_NAME"] for item in top_10_berths_kandla_port_as_per_utilization]
            util_values = [item["Utilization (%)"] for item in top_10_berths_kandla_port_as_per_utilization]


            prompt = f"""
You are a senior data analyst at Kandla Port. You have been given operational data related to port berths.

Your task is to generate clear, helpful, and human-friendly insights from the following data. Focus on identifying **strengths**, **bottlenecks**, and **areas needing attention**, while also suggesting improvements.

📊 **Data Summary**:
- Top 10 Berths by Utilization: {util_labels} with values {util_values}
- Total Berths in Use: {summary['total_berths_on_kandla_port']}
- Top 10 Berths by Usage: {top_10_berths_as_per_usage['labels']} with values {top_10_berths_as_per_usage['values']}
- Average Docking & Undocking Duration (hours): {avg_berth_duration_for_dockUndock_inHrs['labels']} with durations {avg_berth_duration_for_dockUndock_inHrs['values']}
- Berths Idle for 2+ days: {idle_labels} with values {idle_values}

📝 **Instructions**:
- Present your insights as **bullet points** using circle bullets (•), each on a **new line** (`\n`).
- Start each point with a **bold heading** (no stars), and add relevant **emojis** for visual cues.
- For key words like “increase”, “decrease”, “delay”, “maintenance”, or “alert”, briefly **explain the cause or impact** before the keyword.
- Use a **clear, simple tone** that even a non-technical stakeholder can understand.
- Make sure to include:
    - Berths performing **exceptionally well** 🚀
    - Berths showing **delays or inefficiencies** 🐢
    - Possible reasons for long turnaround time ⏱️
    - Any outliers or inconsistencies worth noting 🔍

🔧 **Lastly**, suggest 3-4 **practical improvement ideas** to increase berth efficiency. Write them in bullet points using new lines (`\n`), based on the data.

Output:
"""

        else:
            # Generate port-level insight
            summary = requests.get('http://localhost:5001/api/summary_stats').json()
            monthly_ship_cnt = requests.get('http://localhost:5001/api/monthly_ship_count_and_forecast').json()
            top_5_ship_types = requests.get('http://localhost:5001/api/top_ship_types').json()
            top_10_frequent_ships = requests.get('http://localhost:5001/api/top_frequent_ships').json()
            market_distribution = requests.get('http://localhost:5001/api/market_distribution').json()

            prompt = f"""
You are a senior data analyst at Kandla Port. You are provided with key metrics and traffic data for the port.

Your job is to extract **meaningful**, **easy-to-read** insights that highlight:
- Current port performance 📈
- Major trends 📊
- Bottlenecks or alerts 🚨
- Opportunities for improvement ⚙️

📊 **Port Data**:
- Yearly Ship Count: {summary['total_ships']}
- Monthly Average: {summary['monthly_avg']}
- Active Ships Today: {summary['active_ships']}
- Port Utilization: {summary['port_utilization']}%
- Historical Ship Traffic (Jan 2018 – Apr 2025): {monthly_ship_cnt['historical']['values']}
- Forecasted Ship Traffic (Next 1 year): {monthly_ship_cnt['predicted']['values']}
- Top 5 Ship Types: {top_5_ship_types['labels']} with values {top_5_ship_types['values']}
- Most Frequent Ships (Top 10): {top_10_frequent_ships['labels']} with values {top_10_frequent_ships['values']}
- Market Distribution: {market_distribution['labels']} with values {market_distribution['values']}

📝 **Instructions**:
- Format your insights as bullet points (•), each on a new line (`\n`).
- Start each with a **bold** heading and relevant emojis.
- Highlight any **alerts**, **rising trends**, or **sudden drops** with a short cause or note.
- Keep the language **non-technical** and clear for senior decision-makers.
- Cover:
    - Port traffic trends 📉📈
    - Overutilized or underused resources 📊
    - Consistency or spikes in ship arrivals
    - Any potential risk or delay markers 🛑

🔧 **Finish with 3–4 smart suggestions** to improve port throughput, traffic handling, or resource planning. Write them in bullet points on new lines (`\n`).

Output:
"""

        # Call LLM API
        def generate():
            with requests.post(
                "http://192.168.10.41:11434/api/chat",
                json={
                    "model": "llama3.3:70b",
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": True
                },
                headers={"Content-Type": "application/json"},
                stream=True,
            ) as r:
                for line in r.iter_lines():
                    if line:
                        try:
                            json_data = json.loads(line.decode('utf-8').strip())
                            content_piece = json_data.get("message", {}).get("content", "")
                            if content_piece:
                                yield content_piece
                        except Exception:
                            continue

        return Response(stream_with_context(generate()), content_type='text/plain')

    except Exception as e:
        return Response(f"[ERROR] {str(e)}", content_type='text/plain')


# ************************************************* OLAMA-LLAMA3.1:70b APIs END *************************************************


if __name__ == '__main__':
    app.run(debug=True, port=5001)


