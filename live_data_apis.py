from flask import Flask, jsonify, request
import pandas as pd 
import numpy as np 


app = Flask(__name__)

def load_last_48_hrs_data():
    # Load the data
    df = pd.read_csv('response_portcall_269_simple.csv')
    
# TOTAL SHIPS THAT HAVE VISITED KANDLA PORT IN LAST 48 HRS
@app.route('/api/total_ships_last_48hrs', methods=['GET'])
def total_ships_last_48hrs():

