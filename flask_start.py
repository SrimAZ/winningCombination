from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import matplotlib.pyplot as plt
import operator
from sklearn.decomposition import PCA
import pulp
import re
import stats_graph

app = Flask(__name__)

@app.route("/")
def home():
    #return "Method used: %s" %request.method
    return render_template("mlproj.html")


@app.route("/about")

def about():
    return render_template("projectinfo.html")

@app.route("/playerstats")
def profile():
    return render_template('playerstats.html')

@app.route("/team_selection")
def team_selection():
    return render_template('team_selection.html')

@app.route("/trygraph", methods=["GET","POST"])
def trygraph():
    if(request.method == 'POST'):
        pname = request.form['pname']
        tname = request.form['tname']
        p = stats_graph.gengraph(pname=pname, tname=tname)
        return render_template('showgraph.html', p=p )

@app.route("/batgraph", methods=["GET","POST"])
def batgraph():
    if(request.method == 'POST'):
        pname = request.form['pbatname']
        tname = request.form['battname']
        p = stats_graph.genbatgraph(pname=pname, tname=tname)
        return render_template('showgraph.html', p=p )

@app.route("/generate_stats_batsman", methods=["GET","POST"])
def generate_stats_batsman():
    if(request.method == 'POST'):
        data = pd.read_csv("batsman_name_mvpi.csv")
        return render_template('statspage.html', table=data.to_html())

@app.route("/generate_stats_bowlers", methods=["GET","POST"])
def generate_stats_bowlers():
    if(request.method == 'POST'):
        data = pd.read_csv("bowler_name_mvpi.csv")
        return render_template('statspage.html', table=data.to_html())

@app.route("/generate_team", methods=["GET","POST"])
def generate_team():
    if(request.method == 'POST'):
        nob = request.form['nob']
        nobl = request.form['nobl']
        bat_budget = request.form['bat_budget']
        bowl_budget = request.form['bowl_budget']
        spin_no = request.form['spin_no']
        p = stats_graph.genteam(nob=nob,nobl=nobl,bat_budget=bat_budget,bowl_budget=bowl_budget,spin_no=spin_no)
        data=pd.read_csv("batsman_selected.csv")
        data1=pd.read_csv("bowler_selected.csv")
        return render_template('show_team.html', p=p, table=data.to_html(), table1=data1.to_html())

if __name__ == '__main__':
    app.run(debug=True)

